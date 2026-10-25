"""Evaluate only a complete, source-matched selective experiment on AutoDL."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mine_relay.data import MineDataset, read_ids, group_id
from mine_relay.engine import require_cuda, seed_everything, fingerprint, evaluate
from .model import SelectiveMine
from .runtime import SCHEMA, RECIPE_FIELDS, read, write, require, sources, validate_recipe, validate_resume, validate_history


def arguments(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('run', 'data-root', 'split', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--checkpoint', choices=('best', 'last'), default='best')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--corruption', choices=('none', 'holes', 'dark', 'dropout', 'noise'), default='none')
    parser.add_argument('--corruption-seed', type=int, default=2026)
    parser.add_argument('--official-test', action='store_true')
    return parser.parse_args(argv)


def audit_completion(run, kind, code_hashes):
    """Separate completed-run length from the epoch of the selected checkpoint."""
    run = Path(run)
    completion, status, config = (read(run / name) for name in ('completion.json', 'status.json', 'config.json'))
    validate_recipe(config)
    require(completion['schema'] == SCHEMA and status['state'] == 'complete', 'Run has not completed')
    require(completion['completed_epochs'] == status['completed_epochs'] == config['epochs'], 'Incomplete epoch budget')
    require(completion['config_sha256'] == fingerprint(run / 'config.json') and
            completion['history_sha256'] == fingerprint(run / 'history.jsonl'), 'Run records changed')
    require(completion['source_hashes'] == code_hashes, 'Source differs from completed experiment')
    raw = (run / 'history.jsonl').read_bytes()
    require(raw.endswith(b'\n'), 'Unfinished history write')
    history = [json.loads(line) for line in raw.splitlines()]
    require(len(history) == config['epochs'], 'Missing completed epochs')
    validate_history(history, config, completion['steps_per_epoch'])
    expected = {'best', 'last'} if config['protocol'] == 'development' else {'last'}
    require(set(completion['checkpoints']) == expected and kind in expected, 'Invalid selected checkpoint')
    for name, sha in completion['checkpoints'].items():
        require(fingerprint(run / (name + '.pth')) == sha, 'Completed checkpoint changed: ' + name)
    checkpoint = torch.load(run / (kind + '.pth'), map_location='cpu', weights_only=False)
    validate_resume(checkpoint, config, completion['split_hashes'], code_hashes)
    require(checkpoint['steps_per_epoch'] == completion['steps_per_epoch'], 'Checkpoint epoch size differs')
    require(checkpoint['history'] == history[:checkpoint['next_epoch']], 'Selected checkpoint has a different history')
    selected_epoch = history[-1]['best_epoch'] + 1 if kind == 'best' else config['epochs']
    require(checkpoint['next_epoch'] == selected_epoch, 'Wrong selected epoch')
    return config, completion, checkpoint


def validate_evaluation_split(root, split, checkpoint, official_test, kind):
    ids = read_ids(split)
    require(ids and len(set(ids)) == len(ids), 'Empty or duplicated evaluation split')
    eval_ids = set(ids)
    official_ids = set(read_ids(Path(root) / 'val.txt'))
    config = checkpoint['config']
    if official_test:
        require(eval_ids == official_ids and config['protocol'] == 'final' and kind == 'last',
                'Official testing requires full official split and fixed-epoch final last checkpoint')
        require(checkpoint['next_epoch'] == config['epochs'], 'Final retraining is incomplete')
    else:
        require(not eval_ids & official_ids, 'Official test data require explicit --official-test')
        require(config['protocol'] == 'development' and ids == checkpoint['val_ids'], 'Use the exact fixed development split')
        require(fingerprint(split) == checkpoint['split_hashes']['val'], 'Development split file changed')
    train_ids = set(checkpoint['train_ids'])
    require(not eval_ids & train_ids and not set(map(group_id, eval_ids)) & set(map(group_id, train_ids)),
            'Evaluation location appeared in training')
    return ids


def main():
    args = arguments()
    require_cuda()
    require(args.batch_size > 0 and args.workers >= 0, 'Invalid loader size')
    archive = args.output.with_name(args.output.stem + '.groups.npz')
    require(args.output.suffix == '.json' and not args.output.exists() and not archive.exists(), 'Use a fresh .json output path')
    code_hashes = sources()
    config, completion, checkpoint = audit_completion(args.run, args.checkpoint, code_hashes)
    validate_evaluation_split(args.data_root, args.split, checkpoint, args.official_test, args.checkpoint)
    seed_everything(config['seed'])
    model = SelectiveMine(config['selection'], config['fusion'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.multimodal.configure_training(joint=True)
    model.multimodal.mix.fill_(1.)
    model.cuda().eval()
    loader = DataLoader(MineDataset(args.data_root, args.split, (config['height'], config['width']),
                                    corruption=args.corruption, seed=args.corruption_seed),
                        batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    training = {key: config[key] for key in RECIPE_FIELDS}
    training.update(completed_epochs=completion['completed_epochs'], selected_checkpoint_epoch=checkpoint['next_epoch'],
                    initial_sha256=checkpoint['initial_hash'], train_split_sha256=checkpoint['split_hashes']['train'],
                    val_split_sha256=checkpoint['split_hashes']['val'])
    metadata = {'architecture': 'mine_selective', 'checkpoint_kind': args.checkpoint,
                'checkpoint_sha256': completion['checkpoints'][args.checkpoint],
                'completion_sha256': fingerprint(args.run / 'completion.json'),
                'split_sha256': fingerprint(args.split), 'corruption': args.corruption,
                'corruption_seed': args.corruption_seed, 'protocol': config['protocol'], 'official_test': args.official_test,
                'input_size': [config['height'], config['width']],
                'input_fingerprint_contract': 'RGB float32 HWC [0,1] and uint16 depth after resize/corruption',
                'inference': 'single resize, no TTA, CUDA bf16 autocast', 'source_hashes': code_hashes, 'training': training}
    del checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(metadata, metrics=evaluate(model, loader, archive_path=archive, metadata=metadata),
                  group_archive=archive.name, group_archive_sha256=fingerprint(archive))
    write(args.output, result)
    print(json.dumps(result['metrics']['all']), flush=True)


if __name__ == '__main__':
    main()
