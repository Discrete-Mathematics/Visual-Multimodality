"""Run ONLY on AutoDL. These checks have intentionally not run on the user's PC."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import MineRelay, GeometryRelay
from .data import MineDataset, pack_geometry, corrupt_depth
from .legacy.geometry import FX, FY, CX, CY
from .legacy.hha import MineHHA
from .engine import require_cuda, seed_everything, forward_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--hha", type=Path)
    p.add_argument("--imagenet", type=Path)
    args = p.parse_args()
    require_cuda()
    seed_everything(2026)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                      "cuda": torch.version.cuda}), flush=True)
    block = GeometryRelay(64).cuda()
    x = torch.randn(2, 64, 16, 20, device="cuda")
    geo = torch.randn_like(x)
    zero = torch.zeros(2, 1, 64, 80, device="cuda")
    luma = torch.rand_like(zero)
    # A learned nonzero adapter must still produce zero correction without support.
    torch.nn.init.normal_(block.up.weight, std=0.01)
    correction, state = block(x, geo, zero, luma)
    assert torch.isfinite(correction).all() and torch.count_nonzero(correction) == 0
    assert torch.count_nonzero(state["support"]) == 0
    one = torch.ones_like(zero)
    one[:, :, 20:40, 20:45] = 0
    correction, state = block(x, geo, one, luma)
    correction.square().mean().backward()
    assert block.up.weight.grad is not None and torch.isfinite(block.up.weight.grad).all()
    assert block.query.weight.grad is not None and torch.isfinite(block.query.weight.grad).all()
    assert torch.count_nonzero(block.query.weight.grad) > 0
    depth = np.full((64, 80), 2000, dtype=np.uint16)
    depth[20:35, 20:35] = 0
    damaged = corrupt_depth(depth, np.random.default_rng(1), "dropout")
    packed = pack_geometry(damaged, (FX, FY, CX, CY))
    assert not any(np.count_nonzero(v) for v in packed.values())
    dataset = MineDataset(args.data_root, args.split, size=(480, 640), train=False)
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    assert batch["depth"].shape == (1, 3, 480, 640)
    assert batch["hha"].shape == (1, 4, 480, 640)
    assert torch.isfinite(batch["hha"]).all()
    if args.imagenet:
        model = MineRelay().cuda()
        model.initialize_imagenet(args.imagenet)
        model.eval()
        with torch.no_grad():
            result = forward_batch(model, batch)
        assert result["logits"].shape == (1, 16, 480, 640)
        assert torch.isfinite(result["logits"]).all()
        del model, result
    if args.hha:
        reference = MineHHA().cuda().eval()
        reference.load_trained(args.hha)
        reference.set_mix(10)
        tensors = [batch[k].cuda() for k in ("rgb", "depth", "valid", "luma", "hha")]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected, _ = reference(*tensors)
        del reference
        for mode in ("logits", "inject", "relay"):
            model = MineRelay(mode).cuda().eval()
            model.initialize_hha(args.hha)
            with torch.no_grad():
                actual = forward_batch(model, batch)["logits"]
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
            print(f"PASS zero-initialization equivalence: {mode}", flush=True)
            del model, actual
    print("PASS AutoDL structural preflight; this is not a segmentation accuracy result", flush=True)


if __name__ == "__main__":
    main()
