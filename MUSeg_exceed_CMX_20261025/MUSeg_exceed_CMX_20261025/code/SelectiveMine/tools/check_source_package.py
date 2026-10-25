"""Isolated cloud CPU relocation check; uses disposable fixture weights only."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixture', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert Path('/root/autodl-tmp').is_dir() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert sys.flags.isolated and sys.dont_write_bytecode, 'Use python -I -B'
    package = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(package), str(package / 'tools')]
    manifest = json.loads((package / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert sha(package / name) == expected, name
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available()
    from mine_selective.runtime import sources, RECIPE_FIELDS
    from mine_selective.train import arguments
    from mine_relay.data import MineDataset, read_ids, group_id
    from selective_inference_bundle_20261023 import load_for_inference
    assert sources() == manifest['model_source_hashes']
    train = read_ids(package / 'splits/dev_train.txt')
    val = read_ids(package / 'splits/dev_val.txt')
    assert len(train) == 1281 and len(val) == 314
    assert not set(map(group_id, train)) & set(map(group_id, val))
    for key, path in [('train', 'dev_train.txt'), ('val', 'dev_val.txt')]:
        assert sha(package / 'splits' / path) == manifest['split_hashes'][key]
    recipes = json.loads((package / 'recipes.json').read_text())
    values = {'DATA_ROOT': str(args.data_root), 'TRAIN_SPLIT': str(package / 'splits/dev_train.txt'),
              'VAL_SPLIT': str(package / 'splits/dev_val.txt'), 'RUN_DIR': '/root/autodl-tmp/never_started_package_check',
              'BUDGET_RECORD': '/root/autodl-tmp/never_started_package_check_budget.json',
              'IMAGENET': '/root/autodl-tmp/weights/segformer_b2_backbone_weights.pth'}
    parsed = {}
    for name, recipe in recipes['arms'].items():
        argv = [values.get(x[2:-1], x) if x.startswith('${') and x.endswith('}') else x for x in recipe['arguments']]
        cfg = vars(arguments(argv))
        parsed[name] = {key: cfg[key] for key in RECIPE_FIELDS}
        assert parsed[name] == recipe['expected_recipe'], name
    assert parsed['utility']['selection'] == 'utility' and parsed['fixed_control']['selection'] == 'control'
    differences = {k for k in RECIPE_FIELDS if parsed['utility'][k] != parsed['fixed_control'][k]}
    assert differences == {'selection', 'utility_weight'}
    model, metadata = load_for_inference(args.fixture, allow_software_check=True)
    assert metadata['software_check_only'] and metadata['completed_epochs'] == 2 and not model.training
    assert float(model.multimodal.mix) == 1.
    records = []
    with torch.no_grad():
        for condition in ('none', 'dropout'):
            data = MineDataset(args.data_root, package / 'splits/dev_val.txt', (480, 640),
                               corruption=condition, seed=2026)
            item = data[0]
            output = model(*[item[k].unsqueeze(0) for k in ('rgb', 'depth', 'valid', 'luma', 'hha')])
            assert output['logits'].shape == (1, 16, 480, 640)
            assert torch.isfinite(output['logits']).all()
            assert torch.allclose(output['logits'].exp().sum(1), torch.ones((1, 480, 640)), atol=2e-5)
            if condition == 'dropout':
                assert torch.equal(output['logits'], torch.nn.functional.log_softmax(output['rgb_logits'].float(), 1))
            records.append({'condition': condition, 'sample_id': item['id'], 'finite_normalized_outputs': True,
                            'input_fingerprint': item['sensor_fingerprint']})
    origins = {}
    for name, module in sys.modules.items():
        if name.startswith(('mine_selective', 'mine_relay', 'selective_inference_bundle_20261023')):
            path = Path(module.__file__).resolve()
            assert path.is_relative_to(package), (name, str(path))
            origins[name] = str(path.relative_to(package))
    assert sources() == manifest['model_source_hashes']
    report = {'passed': True, 'cloud_cpu_only': True, 'software_fixture_only': True,
              'package_root': str(package), 'manifest_sha256': sha(package / 'manifest.json'),
              'checked_files': len(manifest['files']), 'model_files': len(sources()),
              'isolated_python': bool(sys.flags.isolated), 'model_module_origins': origins,
              'parsed_training_recipes': parsed, 'fixture_sha256': sha(args.fixture), 'checks': records,
              'environment': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'Pillow')},
              'limits': ['No training was launched; recipe arguments were parsed only.',
                         'External raw data and disposable two-epoch weights were supplied explicitly.',
                         'No formal research weights, full-set GPU parity or fresh-machine dependency installation tested.']}
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'passed': True, 'checked_files': len(manifest['files']), 'module_origins': len(origins)}))


if __name__ == '__main__':
    main()
