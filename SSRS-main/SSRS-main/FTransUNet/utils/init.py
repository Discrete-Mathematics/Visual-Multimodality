import torch
import torch.nn as nn

def init_weights(net, init_type='normal', init_gain=0.02):
    """
    初始化网络参数，适配RDFC-GAN所有模块
    Args:
        net: 待初始化的网络
        init_type: 初始化类型（normal/xavier）
        init_gain: 初始化增益（标准差）
    """
    def init_func(m):
        classname = m.__class__.__name__
        # 卷积层/转置卷积层
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        # 全连接层
        elif classname.find('Linear') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        # BatchNorm层
        elif classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight.data, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    
    net.apply(init_func)
    print(f"Network initialized with {init_type} initialization (gain={init_gain})")