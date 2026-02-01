import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """ASPP 多尺度注意力模块（DeepLabV3+ 核心）"""
    def __init__(self, in_channels=2048, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp_convs = nn.ModuleList()
        for rate in atrous_rates:
            self.aspp_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat1x1 = self.conv1x1(x)
        feats = [feat1x1]
        for aspp_conv in self.aspp_convs:
            feats.append(aspp_conv(x))
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        feats.append(global_feat)
        cat_feat = torch.cat(feats, dim=1)
        return self.conv_cat(cat_feat)

class Decoder(nn.Module):
    """DeepLabV3+ 解码层"""
    def __init__(self, in_channels=256, num_classes=19):
        super(Decoder, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x, low_feat=None):
        # 上采样到1/4尺寸
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        if low_feat is not None:
            low_feat = self.conv_low(low_feat)
            x = torch.cat([x, low_feat], dim=1)
            x = self.conv_cat(x)
        # 上采样到原始尺寸
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        return self.conv_out(x)

class DeepLabV3Plus_Head(nn.Module):
    def __init__(self, in_channels=2048, num_classes=19):
        super(DeepLabV3Plus_Head, self).__init__()
        self.aspp = ASPP(in_channels=in_channels, out_channels=256)
        self.decoder = Decoder(in_channels=256, num_classes=num_classes)

    def forward(self, x, low_feat=None):
        x = self.aspp(x)
        x = self.decoder(x, low_feat)
        return x