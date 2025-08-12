1.分层 Transformer 编码器
# 关键特征：无位置编码 + 多尺度输出
class HierarchicalEncoder(nn.Module):
  def forward(self, x):
      features = []
      for i in range(4):
          x = self.overlap_patch_embed(x)  # 重叠分块
          x = self.transformer_block(x)    # Mix-FFN + 高效注意力
          features.append(x)  # 保存多尺度特征
      return features  # [1/4, 1/8, 1/16, 1/32]

2. 轻量MLP解码器
class AllMLPDecoder(nn.Module):
  def forward(self, features):
      # 1. 1x1卷积统一通道数
      unified = [conv(f) for f in features]

      # 2. 上采样到1/4分辨率
      upsampled = [F.interpolate(u, scale_factor=2**i) 
                  for i,u in enumerate(unified)]
