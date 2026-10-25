"""CUDA trainer with fixed grouped splits and completed-epoch resume state."""

import argparse
import datetime as dt
import json
import math
import platform
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mine_relay.data import MineDataset, read_ids
from mine_relay.engine import (require_cuda, seed_everything, seed_worker, fingerprint, validate_protocol,
                               class_weights, forward_batch, evaluate, save_checkpoint)
from .model import SelectiveMine
from .losses import selective_loss
from .runtime import (SCHEMA, require, read, write, sources, approved_deadline, validate_recipe,
                      validate_resume, persist_history, copy_atomic, acquire_run_lock, recover_committed_files,
                      RECIPE_FIELDS, capture_rng, restore_rng)


def arguments(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('data-root', 'train-split', 'run-dir', 'budget-record'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--val-split', type=Path)
    parser.add_argument('--protocol', choices=('development', 'final'), default='development')
    parser.add_argument('--selection', choices=('control', 'utility'), default='utility')
    parser.add_argument('--fusion', choices=('logits', 'inject', 'relay'), default='relay')
    initial = parser.add_mutually_exclusive_group(required=True)
    initial.add_argument('--imagenet', type=Path)
    initial.add_argument('--resume', type=Path)
    for name, value in (('epochs', 80), ('batch-size', 8), ('workers', 8), ('height', 480), ('width', 640), ('seed', 2026)):
        parser.add_argument('--' + name, type=int, default=value)
    for name, value in (('lr', 6e-5), ('paired-weight', .5), ('rare-crop', 0.), ('rgb-weight', .5),
                        ('fused-weight', .5), ('geo-weight', .2), ('temperature', .5), ('max-hours', 3.), ('hourly-rate', 7.98)):
        parser.add_argument('--' + name, type=float, default=value)
    parser.add_argument('--utility-weight', type=float)
    args = parser.parse_args(argv)
    if args.utility_weight is None:
        args.utility_weight = .05 if args.selection == 'utility' else 0.
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    return args


def learning_rate_factor(progress, epochs):
    warmup = min(2, max(1, epochs // 10))
    if progress < warmup:
        return min(1., (progress + 1) / warmup)
    return .5 * (1 + math.cos(math.pi * (progress - warmup) / max(epochs - warmup, 1)))


def optimizer_for(model, lr):
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            scale = 1. if 'encoder' in name else 4.
            decay = 0. if parameter.ndim <= 1 else .01
            groups.setdefault((scale, decay), []).append(parameter)
    return torch.optim.AdamW([{'params': params, 'lr': lr * scale, 'base_lr': lr * scale, 'weight_decay': decay}
                              for (scale, decay), params in groups.items()])


def main():
    args = arguments()
    require_cuda()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config['architecture'] = 'mine_selective'
    validate_recipe(config)
    deadline, budget_sha = approved_deadline(args.budget_record)
    require(time.time() < deadline - 180, 'Insufficient time in the authorized window')
    validate_protocol(args.data_root, args.train_split, args.val_split, args.protocol, pretrained_hha=False)
    seed_everything(args.seed)
    code_hashes = sources()
    split_hashes = {'train': fingerprint(args.train_split), 'val': fingerprint(args.val_split) if args.val_split else None}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_run_lock(args.run_dir)
    checkpoint = None
    if args.resume:
        require(args.resume == args.run_dir / 'last.pth', 'Resume the authoritative last checkpoint in its run directory')
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        validate_resume(checkpoint, config, split_hashes, code_hashes)
        require(not (args.run_dir / 'completion.json').exists(), 'This run is already complete')
        original_config = read(args.run_dir / 'config.json')
        require(all(original_config[key] == config[key] for key in RECIPE_FIELDS), 'Run configuration changed')
        recover_committed_files(args.run_dir, checkpoint)
        write(args.run_dir / ('resume_' + str(time.time_ns()) + '.json'),
              {'config': config, 'resume_sha256': fingerprint(args.resume), 'budget_sha256': budget_sha})
    else:
        require(not any((args.run_dir / name).exists() for name in ('config.json', 'last.pth', 'history.jsonl', 'completion.json')),
                'Choose a fresh run directory')
        write(args.run_dir / 'config.json', config)
    train = MineDataset(args.data_root, args.train_split, (args.height, args.width), train=True,
                        paired=args.paired_weight > 0, rare_crop=args.rare_crop, seed=args.seed)
    weights, counts = class_weights(args.data_root, train.ids)
    val_ids = read_ids(args.val_split) if args.val_split else []
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
                        drop_last=True, worker_init_fn=seed_worker, generator=generator)
    require(len(loader) > 0, 'Training set smaller than one batch')
    val_loader = DataLoader(MineDataset(args.data_root, args.val_split, (args.height, args.width)),
                           batch_size=args.batch_size, num_workers=args.workers, pin_memory=True) if args.val_split else None
    model = SelectiveMine(args.selection, args.fusion)
    if checkpoint:
        require(checkpoint['train_ids'] == train.ids and checkpoint['val_ids'] == val_ids and
                checkpoint['steps_per_epoch'] == len(loader), 'Resume sample order or epoch size changed')
        require(checkpoint['class_counts'] == counts.tolist(), 'Training class counts changed')
        model.load_state_dict(checkpoint['model'], strict=True)
        model.multimodal.configure_training(joint=True)
        model.multimodal.mix.fill_(1.)
    else:
        model.initialize_imagenet(args.imagenet)
    model.cuda()
    optimizer = optimizer_for(model, args.lr)
    history, best, best_epoch, prior_seconds = [], -1., -1, 0.
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        history, best, best_epoch = checkpoint['history'], checkpoint['best'], checkpoint['best_epoch']
        prior_seconds = checkpoint['elapsed_seconds']
        restore_rng(checkpoint['rng'], generator)
    initial_sha = checkpoint['initial_hash'] if checkpoint else fingerprint(args.imagenet)
    checkpoint = None
    write(args.run_dir / 'class_weights.json', {'counts': counts.tolist(), 'weights': weights.cpu().tolist()})
    write(args.run_dir / 'environment.json', {'python': platform.python_version(), 'torch': str(torch.__version__),
          'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'source_hashes': code_hashes,
          'split_hashes': split_hashes, 'budget_sha256': budget_sha, 'initial_sha256': initial_sha,
          'registered_parameters': sum(p.numel() for p in model.parameters()),
          'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
    started = time.monotonic()
    train_end = min(deadline - 90, time.time() + max(0., args.max_hours * 3600 - prior_seconds) - 90)
    require(time.time() < train_end, 'Cumulative training cap is exhausted')
    reason = 'epochs_complete'
    loss_options = {key: getattr(args, key) for key in ('rgb_weight', 'fused_weight', 'geo_weight', 'utility_weight', 'temperature')}
    status = {'state': 'running', 'completed_epochs': len(history), 'planned_epochs': args.epochs,
              'deadline_utc': dt.datetime.fromtimestamp(deadline, dt.timezone.utc).isoformat(), 'budget_sha256': budget_sha}
    write(args.run_dir / 'status.json', status)
    try:
        for epoch in range(len(history), args.epochs):
            if time.time() >= train_end:
                reason = 'wall_time_cap'
                break
            model.train()
            total_loss, steps = 0., 0
            epoch_start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for step, batch in enumerate(loader):
                factor = learning_rate_factor(epoch + step / len(loader), args.epochs)
                for group in optimizer.param_groups:
                    group['lr'] = group['base_lr'] * factor
                optimizer.zero_grad(set_to_none=True)
                label = batch['label'].cuda(non_blocking=True)
                clean = forward_batch(model, batch)
                loss, _ = selective_loss(clean, label, weights, **loss_options)
                if args.paired_weight > 0:
                    damaged = forward_batch(model, batch, damaged=True)
                    damaged_loss, _ = selective_loss(damaged, label, weights, **loss_options)
                    loss = (loss + args.paired_weight * damaged_loss) / (1 + args.paired_weight)
                require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                total_loss += float(loss.detach())
                if steps % 20 == 0:
                    print(json.dumps({'epoch': epoch + 1, 'step': steps, 'loss': float(loss.detach())}), flush=True)
                if time.time() >= train_end:
                    break
            if steps != len(loader):
                reason = 'wall_time_cap'
                write(args.run_dir / ('discarded_partial_epoch_' + str(time.time_ns()) + '.json'),
                      {'epoch': epoch, 'steps': steps, 'reason': reason,
                       'note': 'No partial weights committed; resume restores the previous completed epoch and its RNG state.'})
                break
            metrics = evaluate(model, val_loader) if val_loader else None
            improved = metrics is not None and metrics['all']['miou'] > best
            if improved:
                best, best_epoch = metrics['all']['miou'], epoch
            elapsed = prior_seconds + time.monotonic() - started
            row = {'epoch': epoch, 'steps': steps, 'partial': False, 'loss': total_loss / steps, 'metrics': metrics,
                   'best': best, 'best_epoch': best_epoch, 'epoch_seconds': time.monotonic() - epoch_start,
                   'elapsed_seconds': elapsed, 'peak_allocated_gb': torch.cuda.max_memory_allocated() / 2 ** 30}
            history.append(row)
            previous_best_sha = fingerprint(args.run_dir / 'best.pth') if not improved and best_epoch >= 0 else None
            payload = {'schema': SCHEMA, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config,
                       'next_epoch': epoch + 1, 'history': history, 'steps_per_epoch': len(loader),
                       'best': best, 'best_epoch': best_epoch, 'previous_best_sha256': previous_best_sha,
                       'split_hashes': split_hashes, 'source_hashes': code_hashes, 'initial_hash': initial_sha,
                       'train_ids': train.ids, 'val_ids': val_ids, 'class_counts': counts.tolist(),
                       'initialization': 'imagenet', 'elapsed_seconds': elapsed,
                       'rng': capture_rng(generator)}
            # last.pth is the commit record. History and a newly selected best
            # can be reconstructed from it if interrupted between file writes.
            save_checkpoint(args.run_dir / 'last.pth', payload)
            if improved:
                copy_atomic(args.run_dir / 'last.pth', args.run_dir / 'best.pth')
            persist_history(args.run_dir / 'history.jsonl', history)
            status.update(completed_epochs=len(history), best=best, best_epoch=best_epoch,
                          elapsed_seconds=elapsed, updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            write(args.run_dir / 'status.json', status)
            print(json.dumps({'epoch': epoch + 1, 'loss': row['loss'], 'miou': metrics['all']['miou'] if metrics else None,
                              'best': best, 'epoch_seconds': row['epoch_seconds']}), flush=True)
        complete = len(history) == args.epochs
        if complete:
            paths = ['last'] + (['best'] if args.protocol == 'development' else [])
            write(args.run_dir / 'completion.json', {'schema': SCHEMA, 'completed_epochs': len(history),
                  'steps_per_epoch': len(loader), 'source_hashes': code_hashes, 'split_hashes': split_hashes,
                  'config_sha256': fingerprint(args.run_dir / 'config.json'),
                  'history_sha256': fingerprint(args.run_dir / 'history.jsonl'),
                  'checkpoints': {name: fingerprint(args.run_dir / (name + '.pth')) for name in paths}})
        status.update(state='complete' if complete else 'incomplete', reason=reason, completed_epochs=len(history),
                      elapsed_seconds=prior_seconds + time.monotonic() - started,
                      note='Stopping this program does not stop instance billing. No automatic poweroff.')
        write(args.run_dir / 'status.json', status)
    except BaseException as error:
        status.update(state='failed', error=repr(error))
        write(args.run_dir / 'status.json', status)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
