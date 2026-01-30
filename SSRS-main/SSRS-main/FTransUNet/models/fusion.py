import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, in_channels=2048, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # 空间注意力
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                       padding=kernel_size//2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward_channel_attention(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_weight = self.sigmoid_channel(avg_out + max_out)
        return x * channel_weight

    def forward_spatial_attention(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_feature = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.sigmoid_spatial(self.conv_spatial(spatial_feature))
        return x * spatial_weight

    def forward(self, x):
        x = self.forward_channel_attention(x)
        x = self.forward_spatial_attention(x)
        return x