# lib/models/discriminator/patch_gan_discriminator.py
import torch.nn as nn
from lib.models.module.conv_norm_act import ConvModule


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator，与 YAML 完全对齐."""
    def __init__(self,
                 in_channels: int,
                 ndf: int = 64,
                 n_layers: int = 3,
                 norm_cfg: dict = None,
                 activation: str = 'LeakyReLU'):
        """
        参数严格对应 YAML：
          in_channels : 输入通道（1 或 3）
          ndf         : 首层通道数（YAML 里没有时可省，默认 64）
          n_layers    : 中间卷积层数（不含首/尾）
          norm_cfg    : 如 {'type': 'BN2d'}
          activation  : 如 'LeakyReLU'
        """
        super().__init__()

        # 默认 norm_cfg
        if norm_cfg is None:
            norm_cfg = dict(type='BN2d')

        # 通道序列：ndf → ndf*2 → ndf*4 → ... → 1
        out_channels = [ndf * (2 ** i) for i in range(n_layers)] + [1]

        # 卷积参数
        kernel_size = [4] * (n_layers + 1)
        stride      = [2] * n_layers + [1]
        padding     = [1] * (n_layers + 1)

        conv_channels = [in_channels] + out_channels
        sequence = []

        for i in range(len(conv_channels) - 1):
            sequence.append(
                ConvModule(
                    in_channels=conv_channels[i],
                    out_channels=conv_channels[i + 1],
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                    padding=padding[i],
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=None if i == 0 or i == len(conv_channels) - 2 else norm_cfg,
                    activation=None if i == len(conv_channels) - 2 else activation
                )
            )
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)