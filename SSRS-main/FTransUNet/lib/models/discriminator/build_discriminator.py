# lib/models/discriminator/build_discriminator.py
from .patch_gan_discriminator import PatchGANDiscriminator


def build_discriminator(cfg_dic):
    """
    完全按 YAML 字段取值，无隐藏默认。
    cfg_dic 示例：
      type: PatchGAN
      in_channels: 1
      norm_cfg:
        type: BN2d
      activation: LeakyReLU
      ndf: 64
      n_layers: 3
    """
    disc_type = cfg_dic.get('type', 'PatchGAN').lower()
    if disc_type != 'patchgan':
        raise NotImplementedError(f'unknown discriminator type: {disc_type}')

    return PatchGANDiscriminator(
        in_channels=cfg_dic['in_channels'],   # YAML 必须给
        ndf=cfg_dic.get('ndf', 64),
        n_layers=cfg_dic.get('n_layers', 3),
        norm_cfg=cfg_dic.get('norm_cfg', dict(type='BN2d')),
        activation=cfg_dic.get('activation', 'LeakyReLU')
    )