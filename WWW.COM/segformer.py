1.分层 Transformer 编码器
# 无位置编码 + 多尺度输出
class HierarchicalEncoder(nn.Module):
    def __init__(self):
        self.stages = nn.ModuleList([
            OverlapPatchEmbed(patch_size=7, stride=4),  # 阶段1 (1/4)
            OverlapPatchEmbed(patch_size=3, stride=2),  # 阶段2 (1/8)
            OverlapPatchEmbed(patch_size=3, stride=2),  # 阶段3 (1/16)
            OverlapPatchEmbed(patch_size=3, stride=2)   # 阶段4 (1/32)
        ])
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=64, heads=1),   # B0配置
            TransformerBlock(dim=128, heads=2),
            TransformerBlock(dim=320, heads=5),
            TransformerBlock(dim=512, heads=8)
        ])
    
    def forward(self, x):
        features = []
        for stage, block in zip(self.stages, self.blocks):
            x = stage(x)        # 重叠分块
            x = block(x)        # Transformer处理
            features.append(x)  # 保存多尺度特征
        return features  # [1/4, 1/8, 1/16, 1/32]
      
2.用Mix-FFN替代传统位置编码
class MixFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*4)
        self.conv = nn.Conv2d(dim*4, dim*4, 3, padding=1, groups=dim*4)  # 深度卷积
        self.fc2 = nn.Linear(dim*4, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.conv(x.permute(0,3,1,2)).permute(0,2,3,1) 
        return self.fc2(x)

3. 高效自注意力
class EfficientAttention(nn.Module):
    def __init__(self, dim, heads, reduction_ratio=8):
        self.reduction = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio),
            nn.Linear(dim // reduction_ratio, dim)
        ) if reduction_ratio > 1 else nn.Identity()

    def forward(self, q, k, v):
        k = self.reduction(k)  # 序列压缩
        v = self.reduction(v)  # R=64→16→4→1

4. 轻量MLP解码器
class MLPDecoder(nn.Module):
    def __init__(self, in_channels=[64, 128, 320, 512], out_ch=256):
        self.projs = nn.ModuleList([
            nn.Conv2d(ch, out_ch, 1) for ch in in_channels
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch * 4, out_ch, 1),  # 多尺度特征融合
            nn.ReLU()
        )
        self.classifier = nn.Conv2d(out_ch, num_classes, 1)
    
    def forward(self, features):
        # 统一通道数并上采样
        decoded = []
        for i, feat in enumerate(features):
            feat = self.projs[i](feat.permute(0,3,1,2))
            feat = F.interpolate(feat, scale_factor=2**i, mode='bilinear')
            decoded.append(feat)
        # 特征融合
        fused = torch.cat(decoded, dim=1)
        return self.classifier(self.fusion(fused))
      
