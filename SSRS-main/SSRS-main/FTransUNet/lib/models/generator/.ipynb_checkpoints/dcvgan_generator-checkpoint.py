# lib/models/generator/dcvgan_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'instance':
        return nn.InstanceNorm2d
    else:
        return lambda x: nn.Identity()


def init_weights(net, init_type='normal', init_gain=0.02):
    for m in net.modules():
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'initialization method {init_type} is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)


class DCVGANGenerator(nn.Module):
    """
    Encoder-Decoder + Skip (U-Net style)
    输入：N×in_channels×H×W  (默认 4: RGB+raw_depth)
    输出：N×1×H×W            (refined depth)
    """

    def __init__(self,
                 in_channels=4,
                 out_channels=1,
                 ngf=64,
                 norm_type='batch',
                 init_type='normal',
                 init_gain=0.02,
                 use_spectral=False):
        super(DCVGANGenerator, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        conv = lambda i, o, k, s, p, bias: \
            spectral_norm(nn.Conv2d(i, o, k, s, p, bias=bias)) if use_spectral \
            else nn.Conv2d(i, o, k, s, p, bias=bias)

        # -------- Encoder --------
        self.e1 = conv(in_channels, ngf, 4, 2, 1, False)          # 1/2
        self.e2 = conv(ngf, ngf * 2, 4, 2, 1, False)              # 1/4
        self.norm2 = norm_layer(ngf * 2)
        self.e3 = conv(ngf * 2, ngf * 4, 4, 2, 1, False)          # 1/8
        self.norm3 = norm_layer(ngf * 4)
        self.e4 = conv(ngf * 4, ngf * 8, 4, 2, 1, False)          # 1/16
        self.norm4 = norm_layer(ngf * 8)
        self.e5 = conv(ngf * 8, ngf * 8, 4, 2, 1, False)          # 1/32
        self.norm5 = norm_layer(ngf * 8)

        # -------- Decoder --------
        self.d5 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1)  # 1/16
        self.dnorm5 = norm_layer(ngf * 8)
        self.d4 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1)  # 1/8
        self.dnorm4 = norm_layer(ngf * 4)
        self.d3 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1)  # 1/4
        self.dnorm3 = norm_layer(ngf * 2)
        self.d2 = nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1)      # 1/2
        self.dnorm2 = norm_layer(ngf)
        self.d1 = nn.ConvTranspose2d(ngf * 2, out_channels, 4, 2, 1)  # 1/1

        # -------- init --------
        init_weights(self, init_type, init_gain)

    def forward(self, rgb, raw_depth, normal=None):
        """
        兼容 RDFCGAN 调用：
        G_A2B(rgb, raw_depth, normal) -> {'pred_depth': tensor}
        """
        if raw_depth.dim() == 3:
            raw_depth = raw_depth.unsqueeze(1)
        x = torch.cat([rgb, raw_depth], dim=1)          # 4 ch

        # Encoder
        e1 = F.leaky_relu(self.e1(x), 0.2)
        e2 = F.leaky_relu(self.norm2(self.e2(e1)), 0.2)
        e3 = F.leaky_relu(self.norm3(self.e3(e2)), 0.2)
        e4 = F.leaky_relu(self.norm4(self.e4(e3)), 0.2)
        e5 = F.leaky_relu(self.norm5(self.e5(e4)), 0.2)

        # Decoder + Skip
        d5 = F.dropout(F.relu(self.dnorm5(self.d5(e5))), 0.5, training=self.training)
        d5 = torch.cat([d5, e4], dim=1)
        d4 = F.dropout(F.relu(self.dnorm4(self.d4(d5))), 0.5, training=self.training)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = F.dropout(F.relu(self.dnorm3(self.d3(d4))), 0.5, training=self.training)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = F.relu(self.dnorm2(self.d2(d3)))
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.d1(d2)  # 1 ch

        return {'pred_depth': torch.tanh(d1)}   # [-1,1]  外部再 scale 到 [0,1] 即可