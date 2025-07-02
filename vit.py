import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(self,
                 image_size=32,  # 输入图像尺寸
                 patch_size=4,  # 分块大小
                 num_classes=10,  # 分类类别数
                 dim=64,  # 嵌入维度
                 depth=3,  # Transformer层数
                 heads=4,  # 注意力头数
                 mlp_dim=128):  # MLP扩展维度
        super().__init__()

        # 1. 图像分块与嵌入层
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2  # 每个patch的像素数 (RGB)

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),  # 同时完成分块和投影
            nn.Flatten(2)  # (B, dim, H/p * W/p)
        )

        # 2. 可学习参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 分类token
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置编码

        # 3. Transformer编码器
        self.transformer = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=0.1,
                activation='gelu',
                batch_first=True  # (B, seq_len, dim)
            )
            for _ in range(depth)
        ])

        # 4. 分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]

        # 分块嵌入 (B, 3, 32, 32) -> (B, dim, 64) -> (B, 64, dim)
        x = self.patch_embed(x).transpose(1, 2)

        # 添加cls_token和位置编码
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed

        # Transformer处理
        x = self.transformer(x)

        # 用cls_token分类
        return self.mlp_head(x[:, 0])


# 测试代码
if __name__ == "__main__":
    # 超参数配置
    config = {
        "image_size": 32,
        "patch_size": 4,
        "num_classes": 10,
        "dim": 64,
        "depth": 3,
        "heads": 4,
        "mlp_dim": 128
    }

    model = ViT(**config)
    dummy_img = torch.randn(2, 3, 32, 32)  # 模拟2张32x32的RGB图像
    output = model(dummy_img)

    print(f"输入尺寸: {dummy_img.shape}")
    print(f"输出尺寸: {output.shape}")  # 应为 (2, 10)
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")