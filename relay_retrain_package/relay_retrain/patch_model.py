"""Runtime controls for Geometry Relay.

The released source files stay unchanged so the inference bundle hash still
matches. These methods replace the class forwards after that check.
"""

import math

_INSTALLED = False


def _relay_body(module, rgb_feature, geo_feature, valid, luma, log_depth, shift, export):
    import torch
    from torch.nn import functional as F

    size = rgb_feature.shape[-2:]
    control = getattr(module, "control", "full")
    if control == "const_valid":
        valid = torch.ones_like(valid)
    offsets = ((0, 0),) if control == "no_neighbor" else module.offsets
    coverage = F.adaptive_avg_pool2d(valid.float(), size)
    brightness = F.adaptive_avg_pool2d(luma.float(), size)
    mean_depth = F.adaptive_avg_pool2d(log_depth.float() * valid.float(), size) / coverage.clamp_min(1e-6)
    query = F.normalize(module.query(rgb_feature).float(), dim=1, eps=1e-6)
    key = F.normalize(module.key(rgb_feature).float(), dim=1, eps=1e-6)
    value = module.value(geo_feature)
    temperature = module.log_temperature.exp().clamp(max=20.0)
    cost = F.softplus(module.distance_cost)
    scores = []
    available = []
    for dy, dx in offsets:
        score = ((query * shift(key, dy, dx)).sum(1, keepdim=True)
                 * temperature - cost * math.hypot(dy, dx))
        both_seen = coverage * shift(coverage, dy, dx)
        score = score - 4.0 * F.softplus(module.depth_cost) * both_seen * (mean_depth - shift(mean_depth, dy, dx)).abs()
        scores.append(score)
        available.append(shift(coverage, dy, dx))
    affinity = torch.softmax(torch.cat(scores, 1), dim=1)
    supported = affinity * torch.cat(available, 1)
    support = supported.sum(1, keepdim=True)
    weights = supported / support.clamp_min(1e-6)
    transported = torch.zeros_like(value)
    for index, (dy, dx) in enumerate(offsets):
        transported = transported + weights[:, index:index + 1].to(value.dtype) * shift(value, dy, dx)
    quality_logit = module.quality(torch.cat([
        value, transported, coverage.to(value.dtype), support.to(value.dtype),
        brightness.to(value.dtype)], 1))
    confidence = support.to(value.dtype) * quality_logit.sigmoid()
    observed = coverage.to(value.dtype) * value
    inferred = (1.0 - coverage.to(value.dtype)) * confidence * transported
    correction = module.up(torch.cat([observed, inferred], 1))
    center = offsets.index((0, 0))
    visible = (support > 1e-4).to(dtype=weights.dtype)
    neighbor = ((1.0 - weights[:, center:center + 1]) * visible).detach() if export else support.new_zeros(1)
    return (
        correction,
        support,
        coverage,
        quality_logit,
        module.recover(transported),
        neighbor,
    )


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import torch
    from torch.utils.checkpoint import checkpoint

    from mine_relay.legacy.dataset import RGB_MEAN, RGB_STD
    from mine_relay.model import GeometryRelay, shift
    from mine_selective.model import SelectiveMine

    original_relay = GeometryRelay.forward
    original_forward = SelectiveMine.forward

    def relay_forward(self, rgb_feature, geo_feature, valid, luma, log_depth=None):
        control = getattr(self, "control", "full")
        export = bool(getattr(self, "export_weights", False))
        if control == "full" and not export:
            return original_relay(self, rgb_feature, geo_feature, valid, luma, log_depth)
        if log_depth is None:
            log_depth = torch.zeros_like(valid)

        def body(rgb_feature, geo_feature, valid, luma, log_depth):
            return _relay_body(self, rgb_feature, geo_feature, valid, luma, log_depth, shift, export)

        if torch.is_grad_enabled() and not export:
            packed = checkpoint(body, rgb_feature, geo_feature, valid, luma, log_depth, use_reentrant=False)
        else:
            packed = body(rgb_feature, geo_feature, valid, luma, log_depth)
        correction, support, coverage, quality_logit, reconstruction, neighbor = packed
        state = {
            "reconstruction": reconstruction,
            "quality_logit": quality_logit,
            "support": support,
            "coverage": coverage,
        }
        if export:
            state["neighbor_mass"] = neighbor
        return correction, state

    def geometry_tensor(module, depth, valid, hha):
        kind = getattr(module, "geo_input", "hha")
        if kind == "hha":
            return hha
        if kind == "none":
            return torch.zeros_like(hha)
        if kind != "raw_depth":
            raise ValueError(kind)
        level = depth[:, :1]
        geo = level.repeat(1, 3, 1, 1)
        mean = geo.new_tensor(RGB_MEAN).view(1, 3, 1, 1)
        std = geo.new_tensor(RGB_STD).view(1, 3, 1, 1)
        geo = (geo - mean) / std * (valid > 0).to(dtype=geo.dtype)
        # The stem's first convolution expects the same four channels as HHA.
        return torch.cat([geo, (valid > 0).to(dtype=geo.dtype)], dim=1)

    def selective_forward(self, rgb, depth, valid, luma, hha):
        if getattr(self, "geo_input", "hha") != "hha":
            hha = geometry_tensor(self, depth, valid, hha)
        return original_forward(self, rgb, depth, valid, luma, hha)

    GeometryRelay.forward = relay_forward
    SelectiveMine.forward = selective_forward
    _INSTALLED = True


def checkpoint_rgb_encoder(model):
    """Recompute RGB-encoder activations on the backward pass.

    Batch size stays 4. The loss and the parameter update are unchanged.
    The relay's output projection is zero in the initialization checkpoint, so
    putting the relay on the graph needs this to fit in 32 GB.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    module = model.multimodal.base.rgb_encoder
    if getattr(module, "_relay_ckpt", False):
        return
    original = module.forward

    def wrapped(x):
        if not torch.is_grad_enabled():
            return original(x)
        return checkpoint(original, x, use_reentrant=False)

    module.forward = wrapped
    module._relay_ckpt = True
