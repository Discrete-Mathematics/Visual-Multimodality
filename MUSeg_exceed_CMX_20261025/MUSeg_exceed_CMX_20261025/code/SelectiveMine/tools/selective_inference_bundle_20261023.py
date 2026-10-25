"""Cloud-only, source-matched SelectiveMine inference export and loading.

This minimal derivative is not resumable. Original complete checkpoints remain
the authority for training, full evaluation and optimizer/RNG recovery.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


SCHEMA = 'selective-inference-bundle-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_digest(state):
    import torch
    h = hashlib.sha256()
    require(bool(state), 'Empty model state')
    for name in sorted(state):
        tensor = state[name]
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu', 'Expected CPU tensors')
        require(not tensor.is_floating_point() or torch.isfinite(tensor).all().item(), 'Nonfinite model tensor')
        h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        h.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def validate_payload(bundle, source_hashes, allow_software_check=False):
    from mine_relay.legacy.labels import CLASS_NAMES
    from mine_selective.runtime import validate_recipe
    require(bundle.get('schema') == SCHEMA and bundle.get('resumable') is False, 'Unsupported inference derivative')
    require('optimizer' not in bundle and 'rng' not in bundle, 'Training state must stay in the original checkpoint')
    require(bundle['source_hashes'] == source_hashes, 'Inference implementation differs from exported source')
    cfg = bundle['config']
    validate_recipe(cfg)
    require(cfg['protocol'] == 'development' and cfg['fusion'] == 'logits', 'This export covers current development logits arms')
    require(bundle['completed_epochs'] == cfg['epochs'] and 1 <= bundle['selected_epoch'] <= bundle['completed_epochs'],
            'Invalid completed/selected epoch identity')
    require(bundle['class_names'] == CLASS_NAMES, 'Class mapping changed')
    require(bundle['inference_settings'] == {'multimodal_mix': 1.0, 'configure_joint': True, 'eval_mode': True,
                                           'input_size': [cfg['height'], cfg['width']],
                                           'input_order': ['rgb', 'depth', 'valid', 'luma', 'hha'],
                                           'geometry': 'camera_up_with_validity_not_standard_HHA'},
            'Inference settings changed')
    if bundle['software_check_only']:
        require(allow_software_check and bundle['completed_epochs'] == 2, 'Software-only weights are not a research model')
    else:
        require(bundle['completed_epochs'] == 120, 'Research export requires the complete 120-epoch run')
    require(tensor_digest(bundle['model']) == bundle['model_tensor_sha256'], 'Model tensor identity changed')


def load_for_inference(path, device='cpu', allow_software_check=False):
    """Call after placing the matching frozen project on sys.path; inputs follow MineDataset."""
    require(Path('/root/autodl-tmp').is_dir(), 'All model execution remains cloud-only')
    require(device in ('cpu', 'cuda'), 'Unsupported inference device')
    import torch
    from mine_selective.runtime import sources
    from mine_selective.model import SelectiveMine
    manifest = json.loads(Path(path).with_name('export.json').read_text())
    require(manifest['state'] == 'verified_tensor_export' and manifest['file'] == Path(path).name and
            manifest['sha256'] == digest(path), 'Export manifest or file identity differs')
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    require(all(manifest.get(k) == v for k, v in bundle.items() if k != 'model'), 'Bundle metadata differs from export manifest')
    validate_payload(bundle, sources(), allow_software_check)
    cfg = bundle['config']
    model = SelectiveMine(cfg['selection'], cfg['fusion'])
    model.load_state_dict(bundle['model'], strict=True)
    model.multimodal.configure_training(joint=True)
    # mix is deliberately nonpersistent in the original model state dictionary.
    model.multimodal.mix.fill_(bundle['inference_settings']['multimodal_mix'])
    model.to(device).eval()
    return model, {key: value for key, value in bundle.items() if key != 'model'}


def export_run(project, run, output, software_check=False):
    require(Path('/root/autodl-tmp').is_dir() and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'Export on AutoDL CPU with GPU hidden')
    require(not output.exists(), 'Use a fresh export directory')
    sys.path.insert(0, str(project.resolve()))
    import torch
    from mine_selective.runtime import sources
    from mine_selective.evaluate import audit_completion
    from mine_relay.legacy.labels import CLASS_NAMES
    torch.set_num_threads(1)
    require(not torch.cuda.is_available(), 'Export must not use the active training GPU')
    source_hashes = sources()
    cfg, completion, checkpoint = audit_completion(run, 'best', source_hashes)
    if software_check:
        require(run.name == 'disposable_timing_2epochs' and completion['completed_epochs'] == 2,
                'Software export is restricted to the disposable timing fixture')
    else:
        require(completion['completed_epochs'] == 120, 'Do not publish an incomplete research run')
    source = run / 'best.pth'
    source_sha = digest(source)
    require(source_sha == completion['checkpoints']['best'], 'Original checkpoint changed')
    bundle = {'schema': SCHEMA, 'resumable': False, 'software_check_only': bool(software_check),
              'model': checkpoint['model'], 'model_tensor_sha256': tensor_digest(checkpoint['model']),
              'config': cfg, 'source_hashes': source_hashes, 'split_hashes': checkpoint['split_hashes'],
              'initial_sha256': checkpoint['initial_hash'], 'class_names': CLASS_NAMES,
              'completed_epochs': completion['completed_epochs'], 'selected_epoch': checkpoint['next_epoch'],
              'source_checkpoint_sha256': source_sha, 'source_completion_sha256': digest(run / 'completion.json'),
              'source_history_sha256': completion['history_sha256'],
              'inference_settings': {'multimodal_mix': 1.0, 'configure_joint': True, 'eval_mode': True,
                                     'input_size': [cfg['height'], cfg['width']],
                                     'input_order': ['rgb', 'depth', 'valid', 'luma', 'hha'],
                                     'geometry': 'camera_up_with_validity_not_standard_HHA'}}
    validate_payload(bundle, source_hashes, allow_software_check=software_check)
    output.mkdir(parents=True)
    filename = 'software_check_only.pth' if software_check else 'best_inference.pth'
    temp, target = output / (filename + '.tmp'), output / filename
    torch.save(bundle, temp)
    restored = torch.load(temp, map_location='cpu', weights_only=True)
    validate_payload(restored, source_hashes, allow_software_check=software_check)
    require(set(restored['model']) == set(checkpoint['model']), 'Export state keys changed')
    for name, original in checkpoint['model'].items():
        actual = restored['model'][name]
        require(original.shape == actual.shape and original.dtype == actual.dtype and torch.equal(original, actual),
                'Export changed a model tensor: ' + name)
    require({k: v for k, v in restored.items() if k != 'model'} == {k: v for k, v in bundle.items() if k != 'model'},
            'Export changed metadata')
    require(digest(source) == source_sha and sources() == source_hashes, 'Original weights or source changed')
    require(digest(run / 'completion.json') == bundle['source_completion_sha256'] and
            digest(run / 'history.jsonl') == bundle['source_history_sha256'], 'Completed run records changed')
    temp.replace(target)
    report = {k: v for k, v in bundle.items() if k != 'model'}
    report.update(state='verified_tensor_export', file=filename, bytes=target.stat().st_size, sha256=digest(target),
                  every_model_tensor_exact=True, source_checkpoint_bytes=source.stat().st_size,
                  export_script_sha256=digest(__file__),
                  not_included=['optimizer', 'random_states', 'training_history', 'full training/validation ID lists'],
                  limits=['Keep original full checkpoints and records; this derivative cannot resume training.',
                          'Tensor equality alone is not end-to-end GPU evaluation parity.',
                          'One-seed development-selected model, not a final official-test release.'])
    with (output / 'export.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    return report


def main():
    p = argparse.ArgumentParser()
    for key in ('project', 'run', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--software-check', action='store_true')
    args = p.parse_args()
    report = export_run(args.project, args.run, args.output, args.software_check)
    print(json.dumps({k: report[k] for k in ['state', 'file', 'bytes', 'sha256', 'completed_epochs',
                                            'selected_epoch', 'software_check_only']}), flush=True)


if __name__ == '__main__':
    main()
