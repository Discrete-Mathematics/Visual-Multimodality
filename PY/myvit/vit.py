import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """通过卷积核将图像划分为图像块,并嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape # 输入张量x形状, B:batch大小，N:序列长度(就是图像块数量)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # Q、K、V
        scores = q @ k.transpose(-2, -1)                    # 计算点积 Q * K^T
        scores = scores * self.scale                        # 乘以缩放因子
        scores = scores.softmax(dim=-1)                     # softmax归一化, 得到注意力权重
        x = (scores @ v).transpose(1, 2).reshape(B, N, C)   # 权重与值(V)相乘，然后转置回原形状
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """简单的前馈网络
    输入x → 线性变换 → GELU激活 → 线性变换 → 输出
    增强模型的表达能力
    """

    def __init__(self, embed_dim=768, hidden_dim=3072):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """编码器块"""
    def __init__(self, embed_dim=768, num_heads=12, hidden_dim=3072):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)                    
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, hidden_dim)
        
    def forward(self, x):
        norm_x = self.norm1(x)          # 对输入 x 进行LayerNorm归一化
        attn_out = self.attn(norm_x)    # 多头注意力机制
        x = x + attn_out                # 残差连接, 原始输入和注意力输出合并, 对应论文中编码器中间的那步

        # 编码器的第二个子层 (sub-layers)
        norm_x2 = self.norm2(x)         
        mlp_out = self.mlp(norm_x2)     # 前馈网络
        x = x + mlp_out                 # 残差连接, 原始输入和前馈网络输出合并
        return x


class ViT(nn.Module):
    """VIT网络结构"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, hidden_dim=3072):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 可位置编码和类别标记
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # 编码器层
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]                                  # 获取batch    
        x = self.patch_embed(x)                         # 图像块嵌入为向量
        cls_tokens = self.cls_token.expand(B, -1, -1)   
        x = torch.cat((cls_tokens, x), dim=1)           
        x = x + self.pos_embed                          # 和位置编码相加
        
        for block in self.blocks:
            x = block(x)
            
        # 分类
        x = self.norm(x)
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)


def create_vit(num_classes=5):
    return ViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        hidden_dim=3072
    )
