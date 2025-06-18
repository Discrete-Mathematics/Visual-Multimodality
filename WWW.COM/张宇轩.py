import torch
import torch.nn as nn

class SimpleViT(nn.Module):
    def __init__(self, image_size=32, patch_size=8, num_classes=10, dim=64):
        super().__init__()
        # 图像分块处理
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2  # 每个patch的像素值 (C×H×W)
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.attention = nn.MultiheadAttention(dim, num_heads=1, batch_first=True)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        # 图像分块 (B, C, H, W) - (B, num_patches, patch_dim)
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 4, 5, 1).contiguous()
        x = x.view(B, -1, C * self.patch_size ** 2)
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x += self.position_embedding
        attn_output, _ = self.attention(x, x, x)
        cls_output = attn_output[:, 0]
        return self.classifier(cls_output)


# 测试代码
if __name__ == "__main__":
    model = SimpleViT(image_size=32, patch_size=8)
    dummy_image = torch.randn(2, 3, 32, 32)
    output = model(dummy_image)
    print("Output shape:", output.shape)  # 应为 (2, 10)
