# lib/models/generator/build_generator.py
import torch.nn as nn

# 假设你把 DCVGANGenerator 定义在同目录下
from .dcvgan_generator import DCVGANGenerator


def build_generator(cfg_dic):
    """
    cfg_dic 即 args.model.G_A2B 或 G_B2A 对应的 dict，例如：
    {
      'type': 'DCVGAN',
      'in_channels': 4,
      'out_channels': 1,
      'ngf': 64,
      'norm': 'batch',
      'init_type': 'normal',
      'init_gain': 0.02
    }
    若某项缺失，用默认值。
    """
    gen_type = cfg_dic.get('type', 'DCVGAN').lower()

    if gen_type == 'dcvgan':
        # 超参
        in_ch     = cfg_dic.get('in_channels', 4)   # RGB + raw_depth (+ normal)
        out_ch    = cfg_dic.get('out_channels', 1)  # refined_depth
        ngf       = cfg_dic.get('ngf', 64)          # 首层通道数
        norm_type = cfg_dic.get('norm', 'batch')    # batch / instance / none
        init_type = cfg_dic.get('init_type', 'normal')
        init_gain = cfg_dic.get('init_gain', 0.02)

        gen = DCVGANGenerator(in_channels=in_ch,
                              out_channels=out_ch,
                              ngf=ngf,
                              norm_type=norm_type,
                              init_type=init_type,
                              init_gain=init_gain)
        return gen

    else:
        raise NotImplementedError(f'unknown generator type: {gen_type}')