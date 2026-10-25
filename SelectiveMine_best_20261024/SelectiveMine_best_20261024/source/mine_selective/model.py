"""Independent appearance supervision and learned selection between predictions.

This candidate reuses the frozen MineRelay feature extractors and fusion path.
It is not a demonstrated improvement or a claim of novel modality gating.
"""

import copy
import torch
from torch import nn
from torch.nn import functional as F

from mine_relay.model import MineRelay


class SelectiveMine(nn.Module):
    """Same experts in both controls; utility mode learns a spatial mixing weight.

    The RGB encoder is shared but the appearance decoder is independent of the
    fusion decoder. Missing depth selects the explicitly supervised RGB head.
    This ensures a defined fallback, not a lower bound on segmentation accuracy.
    """

    def __init__(self, selection='utility', fusion='relay', hidden=32):
        super().__init__()
        if selection not in ('control', 'utility'):
            raise ValueError('selection must be control or utility')
        if fusion not in ('logits', 'inject', 'relay'):
            raise ValueError('Unknown fusion control')
        self.selection, self.fusion = selection, fusion
        self.multimodal = MineRelay(fusion)
        self.appearance_decoder = copy.deepcopy(self.multimodal.base.decoder)
        # Probabilities of both experts, their disagreement, entropy, coverage,
        # and brightness. No ground truth enters the router's forward inputs.
        self.router = nn.Sequential(nn.Conv2d(52, hidden, 1), nn.GELU(),
                                    nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
                                    nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, 2.)
        for p in self.router.parameters():
            p.requires_grad = selection == 'utility'

    def initialize_imagenet(self, path):
        self.multimodal.initialize_imagenet(path)

    def forward(self, rgb, depth, valid, luma, hha):
        net = self.multimodal
        rgb_features = net.base.rgb_encoder(rgb)
        depth_features = net.base.depth_encoder(depth)
        fused = [gate(r, d, valid, luma)[0] for gate, r, d in zip(net.base.gates, rgb_features, depth_features)]
        geometry = net.geo_encoder(net.stem(hha))
        mixed, relay_state = [], []
        for i, (feature, geo) in enumerate(zip(fused, geometry)):
            if geo.shape[-2:] != feature.shape[-2:]:
                geo = F.interpolate(geo, feature.shape[-2:], mode='bilinear', align_corners=False)
            if self.fusion == 'relay':
                delta, state = net.relay[i](feature, geo, valid, luma, depth[:, :1])
                relay_state.append(state)
            elif self.fusion == 'inject':
                delta = net.inject[i](geo)
            else:
                delta = 0.
            mixed.append(feature + delta)
        size = rgb.shape[-2:]
        interpolate = lambda x: F.interpolate(x, size, mode='bilinear', align_corners=False)
        base_logits = interpolate(net.base.decoder(mixed))
        geo_logits = interpolate(net.geo_decoder(geometry))
        present = (valid.flatten(1).sum(1) > 0).view(-1, 1, 1, 1)
        fused_logits = base_logits + torch.tanh(net.open_raw) * net.mix * present.to(base_logits.dtype) * geo_logits
        rgb_logits = interpolate(self.appearance_decoder(rgb_features))
        log_rgb = F.log_softmax(rgb_logits.float(), dim=1)
        log_fused = F.log_softmax(fused_logits.float(), dim=1)
        # Expert representations are detached only for router inputs, not for
        # the final mixture loss, which still trains both expert predictions.
        rgb_p, fused_p = log_rgb.detach().exp(), log_fused.detach().exp()
        entropy_rgb = -(rgb_p * log_rgb.detach()).sum(1, keepdim=True)
        entropy_fused = -(fused_p * log_fused.detach()).sum(1, keepdim=True)
        grid = rgb_features[0].shape[-2:]
        evidence = torch.cat([rgb_p, fused_p, (rgb_p - fused_p).abs(), entropy_rgb, entropy_fused,
                              valid.float(), luma.float()], dim=1)
        router_logit = interpolate(self.router(F.adaptive_avg_pool2d(evidence, grid)))
        if self.selection == 'utility':
            mixture = torch.logaddexp(log_rgb + F.logsigmoid(-router_logit.float()),
                                     log_fused + F.logsigmoid(router_logit.float()))
            log_probability = torch.where(present, mixture, log_rgb)
            effective_gate = torch.where(present, router_logit.float().sigmoid(), 0.)
        else:
            log_probability = torch.where(present, log_fused, log_rgb)
            effective_gate = present.to(log_probability.dtype).expand_as(router_logit)
        return {'logits': log_probability, 'rgb_logits': rgb_logits, 'fused_logits': fused_logits,
                'geo_logits': geo_logits, 'sensor_present': present.to(base_logits.dtype),
                'router_logit': router_logit, 'geometry': geometry, 'relay': relay_state,
                'effective_gate': effective_gate, 'selection': self.selection}
