import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    """
    经典CBAM注意力模块（Convolutional Block Attention Module）
    适配阶段2：输入通道数2048，输出通道数2048，支持跨模态特征融合细化
    结构：通道注意力（CAM）→ 空间注意力（SAM），依次对特征进行加权优化
    """
    def __init__(self, in_channels=2048, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio  # 通道注意力的压缩比，默认16（经典值）
        self.kernel_size = kernel_size  # 空间注意力的卷积核尺寸，默认7（经典值）

        # 1. 通道注意力模块（CAM）：聚焦"哪些通道的特征更重要"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        # 压缩-激励结构：全连接层（简化为1x1卷积，更高效适配卷积特征）
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # 2. 空间注意力模块（SAM）：聚焦"哪些空间位置的特征更重要"
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                                       padding=kernel_size//2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward_channel_attention(self, x):
        """
        通道注意力前向传播
        x: 输入特征图，形状 [B, C, H, W]（B=批次，C=通道，H=高，W=宽）
        """
        # 全局池化：[B, C, H, W] → [B, C, 1, 1]
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        
        # 通道权重融合：[B, C, 1, 1]
        channel_weight = self.sigmoid_channel(avg_out + max_out)
        
        # 通道加权：输入特征 × 通道权重
        return x * channel_weight

    def forward_spatial_attention(self, x):
        """
        空间注意力前向传播
        x: 经过通道注意力加权后的特征图，形状 [B, C, H, W]
        """
        # 空间维度池化：[B, C, H, W] → [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道维度平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道维度最大池化
        
        # 拼接空间特征：[B, 2, H, W]
        spatial_feature = torch.cat([avg_out, max_out], dim=1)
        
        # 空间权重预测：[B, 2, H, W] → [B, 1, H, W]
        spatial_weight = self.sigmoid_spatial(self.conv_spatial(spatial_feature))
        
        # 空间加权：输入特征 × 空间权重
        return x * spatial_weight

    def forward(self, x):
        """
        CBAM完整前向传播：通道注意力 → 空间注意力
        输入：
            x: 双模态融合后的特征图（或单模态特征图），形状 [B, 2048, H, W]
        输出：
            经过通道+空间注意力加权的细化特征图，形状 [B, 2048, H, W]
        """
        # 步骤1：通道注意力加权
        x = self.forward_channel_attention(x)
        
        # 步骤2：空间注意力加权
        x = self.forward_spatial_attention(x)
        
        # 步骤3：返回细化后的特征图（通道数保持2048，匹配后续分割头输入）
        return x