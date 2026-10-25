"""Experiment identity and deadline helpers for the selective candidate."""

import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import random

import numpy as np
import torch

from mine_relay.engine import fingerprint, source_fingerprints

SCHEMA = 'mine-selective-training-v1'
RECIPE_FIELDS = ('protocol', 'selection', 'fusion', 'seed', 'epochs', 'batch_size', 'workers',
                 'height', 'width', 'lr', 'paired_weight', 'rare_crop', 'rgb_weight', 'fused_weight',
                 'geo_weight', 'utility_weight', 'temperature')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def copy_atomic(source, target):
    target = Path(target)
    temp = target.with_suffix(target.suffix + '.tmp')
    shutil.copyfile(source, temp)
    temp.replace(target)


def sources():
    result = {'mine_relay/' + key: value for key, value in source_fingerprints().items()}
    result.update({'mine_selective/' + p.name: fingerprint(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    return result


def approved_deadline(path):
    """Validate the recorded extension arithmetic; never start a new budget clock."""
    data = read(path)
    require(bool(data.get('user_authorization')), 'Missing recorded authorization')
    initial = dt.datetime.fromisoformat(data['initial_started_utc'].replace('Z', '+00:00'))
    first = dt.datetime.fromisoformat(data['initial_deadline_utc'])
    last = dt.datetime.fromisoformat(data['deadline_utc'])
    require(all(value.utcoffset() == dt.timedelta(0) for value in (initial, first, last)), 'Budget must use UTC timestamps')
    hours = [data[key] for key in ('initial_budget_hours', 'authorized_additional_hours', 'total_budget_hours')]
    require(all(math.isfinite(value) and value >= 0 for value in hours) and hours[2] > 0, 'Invalid budget hours')
    require(hours[0] + hours[1] == hours[2], 'Budget hours do not add up')
    require(abs((first - initial).total_seconds() - hours[0] * 3600) < 1 and
            abs((last - initial).total_seconds() - hours[2] * 3600) < 1, 'Budget anchor was changed')
    return last.timestamp(), fingerprint(path)


def validate_recipe(config):
    require(config['protocol'] in ('development', 'final'), 'Only clean development or fixed final retraining is supported')
    require(config['selection'] in ('control', 'utility') and config['fusion'] in ('logits', 'inject', 'relay'), 'Unknown architecture')
    require(config['epochs'] >= 1 and config['batch_size'] >= 1 and config['workers'] >= 0, 'Invalid training size')
    require(config['height'] >= 32 and config['width'] >= 32, 'Input grid is too small')
    for key in ('lr', 'temperature', 'max_hours', 'hourly_rate'):
        require(math.isfinite(config[key]) and config[key] > 0, 'Invalid ' + key)
    for key in ('paired_weight', 'rgb_weight', 'fused_weight', 'geo_weight', 'utility_weight'):
        require(math.isfinite(config[key]) and config[key] >= 0, 'Invalid ' + key)
    require(0 <= config['rare_crop'] <= 1, 'Invalid rare-crop probability')
    require(config['selection'] != 'control' or config['utility_weight'] == 0, 'Fixed control cannot use utility supervision')


def validate_resume(checkpoint, config, split_hashes, code_hashes):
    require(checkpoint.get('schema') == SCHEMA and checkpoint.get('initialization') == 'imagenet', 'Unsupported checkpoint')
    require(checkpoint['source_hashes'] == code_hashes, 'Source changed; start a named new experiment')
    require(checkpoint['split_hashes'] == split_hashes, 'Resume data split changed')
    for key in RECIPE_FIELDS:
        require(checkpoint['config'][key] == config[key], 'Resume recipe changed: ' + key)
    history = checkpoint['history']
    require(len(history) == checkpoint['next_epoch'] and len(history) <= config['epochs'], 'Incomplete checkpoint history')
    validate_history(history, config, checkpoint['steps_per_epoch'])
    require(checkpoint['best'] == history[-1]['best'] and checkpoint['best_epoch'] == history[-1]['best_epoch'],
            'Checkpoint selection differs from history')
    require(math.isfinite(checkpoint['elapsed_seconds']) and checkpoint['elapsed_seconds'] >= 0, 'Invalid elapsed time')


def validate_history(history, config, steps_per_epoch):
    require(history and steps_per_epoch > 0, 'No completed training')
    require([row['epoch'] for row in history] == list(range(len(history))), 'Nonconsecutive completed epochs')
    require(all(not row['partial'] and row['steps'] == steps_per_epoch for row in history), 'Partial epoch checkpoint rejected')
    require(all(math.isfinite(row['loss']) for row in history), 'Invalid recorded loss')
    best, best_epoch = -1., -1
    for row in history:
        if config['protocol'] == 'development':
            score = row['metrics']['all']['miou']
            require(math.isfinite(score) and 0 <= score <= 1, 'Invalid validation mIoU')
            if score > best:
                best, best_epoch = score, row['epoch']
        else:
            require(row['metrics'] is None, 'Final retraining must not select on validation or test data')
        require(row['best'] == best and row['best_epoch'] == best_epoch, 'History selection is inconsistent')


def persist_history(path, history):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in history), encoding='utf-8')
    temp.replace(path)


def recover_committed_files(run, checkpoint):
    """Call only after validating last.pth, the authoritative epoch commit."""
    run = Path(run)
    if checkpoint['config']['protocol'] == 'development':
        if checkpoint['best_epoch'] == checkpoint['next_epoch'] - 1:
            copy_atomic(run / 'last.pth', run / 'best.pth')
        else:
            require((run / 'best.pth').exists() and
                    fingerprint(run / 'best.pth') == checkpoint['previous_best_sha256'],
                    'Prior selected checkpoint is missing or changed')
    persist_history(run / 'history.jsonl', checkpoint['history'])


def capture_rng(generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all(), 'loader': generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['loader'])


def acquire_run_lock(run):
    # Kernel lock releases on process exit. A stale PID text file is not a lock.
    import fcntl
    handle = (Path(run) / 'training.lock').open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('Another trainer already owns this run')
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle
