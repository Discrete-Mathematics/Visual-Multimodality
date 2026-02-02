
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install opencv-python-headless scikit-learn scikit-image albumentations
!pip install einops timm

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from google.colab import drive
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import json
from tqdm import tqdm
import math


drive.mount('/content/drive')

DATASET_PATH = '/content/drive/MyDrive/MUSeg'
TRAIN_RGB_DIR = os.path.join(DATASET_PATH, 'RGB')
TRAIN_LABEL_DIR = os.path.join(DATASET_PATH, 'Label')


print("检查数据集划分文件...")
for split_file in ['train.txt', 'val.txt']:
    file_path = os.path.join(DATASET_PATH, split_file)

# 使用现有划分文件
SPLIT_FILES = {
    'train': os.path.join(DATASET_PATH, 'train.txt'),
    'val': os.path.join(DATASET_PATH, 'val.txt')
}


class MUSegDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, image_size=(224, 224)):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size

        if split not in SPLIT_FILES:
            raise ValueError(f"无效的数据集划分: {split}，可用选项: {list(SPLIT_FILES.keys())}")


        split_file = SPLIT_FILES[split]
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"划分文件不存在: {split_file}")

        with open(split_file, 'r') as f:
            self.img_ids = [line.strip() for line in f.readlines() if line.strip()]

        print(f"{split} 数据集: {len(self.img_ids)} 个样本")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        rgb_path = os.path.join(TRAIN_RGB_DIR, f"{img_id}.jpg")
        label_path = os.path.join(TRAIN_LABEL_DIR, f"{img_id}_label.png")

        image = cv2.imread(rgb_path)
        if image is None:
            raise ValueError(f"无法读取图像: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"无法读取标签: {label_path}")

        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']

            label = augmented['mask'].long()
        else:

            image = cv2.resize(image, self.image_size)
            label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(image.transpose(2,0,1)).float() / 255.0
            label = torch.from_numpy(label).long()

    
        if label.ndim > 2:
            label = label.squeeze()

        label = (label > 0).long()

        return image, label


image_size = (224, 224)
train_transform = A.Compose([
    A.Resize(*image_size),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(*image_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

try:
    train_dataset = MUSegDataset(DATASET_PATH, split='train', transform=train_transform)
    val_dataset = MUSegDataset(DATASET_PATH, split='val', transform=val_transform)

  
    test_dataset = val_dataset

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = val_loader  # 使用验证集作为测试集

    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")
    print(f"测试集: 使用验证集 ({len(val_dataset)} 张)")


    valid_data = True
    for i in range(min(len(train_dataset), 50)):
        _, lbl = train_dataset[i]
        if lbl.max() > 1 or lbl.min() < 0:
            print(f"错误：样本 {i} 标签值异常: min={lbl.min()}, max={lbl.max()}")
            valid_data = False
            break


    if len(all_images) > 0:
        
        np.random.shuffle(all_images)
        split_idx = int(0.8 * len(all_images))
        train_ids = all_images[:split_idx]
        val_ids = all_images[split_idx:]

        def save_split_file(ids, filename):
            with open(os.path.join(DATASET_PATH, filename), 'w') as f:
                for img_id in ids:
                    f.write(f"{img_id}\n")

        save_split_file(train_ids, 'train.txt')
        save_split_file(val_ids, 'val.txt'))

        train_dataset = MUSegDataset(DATASET_PATH, split='train', transform=train_transform)
        val_dataset = MUSegDataset(DATASET_PATH, split='val', transform=val_transform)
        test_dataset = val_dataset

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = val_loader
    else:
        raise ValueError("未找到任何图像文件")


class SimpleAgentAttention(nn.Module):
    """简化的代理注意力模块 - 修正版"""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.num_agents = 16
        self.agent_conv = nn.Conv2d(in_channels, self.num_agents, kernel_size=1)

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        agent_logits = self.agent_conv(x).view(B, self.num_agents, N)
        agent_weights = self.softmax(agent_logits) # 在空间维度上归一化

     
        value = self.value_conv(x).view(B, C, N)  # [B, C, H*W]

        
        aggregated_agents = torch.bmm(agent_weights, value.transpose(1, 2))

        query = self.query_conv(x).view(B, C, N)  # [B, C, H*W]

        
        broadcast_scores = torch.bmm(query.transpose(1, 2), aggregated_agents.transpose(1, 2)) / (C ** 0.5)
        broadcast_weights = self.softmax(broadcast_scores)  # 在代理维度上归一化

        broadcast_features = torch.bmm(broadcast_weights, aggregated_agents)
        broadcast_features = broadcast_features.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]

        local_features = self.depthwise_conv(x)
        local_features = self.pointwise_conv(local_features)

      
        combined = broadcast_features + local_features

  
        output = self.output_proj(combined)

      
        output = output + x

        return output


class SimplifiedSwinBlock(nn.Module):
    """简化的Swin Transformer块 - 避免复杂窗口操作"""
    def __init__(self, in_channels, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.in_channels = in_channels

        # 层归一化
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        # 多头自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True
        )

        # MLP
        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, in_channels),
            nn.Dropout(drop)
        )

        # Dropout
        self.dropout = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, L, C = x.shape

        # 保存残差连接
        shortcut = x

        # 第一层归一化 + 注意力
        x_norm = self.norm1(x)
        x_attn, _ = self.attention(x_norm, x_norm, x_norm)
        x = shortcut + self.dropout(x_attn)

        # 第二层归一化 + MLP
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = x + self.dropout(x_mlp)

        return x


class SimplePatchMerging(nn.Module):
    """简化的下采样层"""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2

        # 使用卷积实现下采样
        self.conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(self.out_channels)

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = H

        # 重塑为2D
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)

        # 下采样
        x_down = self.conv(x_2d)  # [B, C*2, H/2, W/2]

        # 展平并归一化
        B, C_new, H_new, W_new = x_down.shape
        x_down = x_down.flatten(2).transpose(1, 2)  # [B, H_new*W_new, C_new]
        x_down = self.norm(x_down)

        return x_down, H_new, W_new

class SimplePatchExpanding(nn.Module):
    """简化的上采样层"""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // 2

        # 使用转置卷积实现上采样
        self.conv = nn.ConvTranspose2d(in_channels, self.out_channels, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(self.out_channels)

    def forward(self, x, H, W):
        # x: [B, L, C]
        B, L, C = x.shape

        # 重塑为2D
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)

        # 上采样
        x_up = self.conv(x_2d)  # [B, C//2, H*2, W*2]

        # 展平并归一化
        B, C_new, H_new, W_new = x_up.shape
        x_up = x_up.flatten(2).transpose(1, 2)  # [B, H_new*W_new, C_new]
        x_up = self.norm(x_up)

        return x_up, H_new, W_new


class SimpleAgentSwinUNet(nn.Module):
    def __init__(self, num_classes=2, img_size=224):
        super().__init__()

        self.img_size = img_size
        self.patch_size = 4
        self.embed_dim = 96

        # 补丁嵌入
        self.patch_embed = nn.Conv2d(
            3, self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )


        self.pos_embed = nn.Parameter(
            torch.zeros(1, (img_size // self.patch_size) ** 2, self.embed_dim)
        )


        self.encoder_stage1 = self._make_encoder_stage(self.embed_dim, 2, 3)  # 96, 56x56
        self.encoder_stage2 = self._make_encoder_stage(self.embed_dim * 2, 2, 6)  # 192, 28x28
        self.encoder_stage3 = self._make_encoder_stage(self.embed_dim * 4, 6, 12)  # 384, 14x14
        self.encoder_stage4 = self._make_encoder_stage(self.embed_dim * 8, 2, 24)  # 768, 7x7

        self.downsample1 = SimplePatchMerging(self.embed_dim)  # 96 -> 192
        self.downsample2 = SimplePatchMerging(self.embed_dim * 2)  # 192 -> 384
        self.downsample3 = SimplePatchMerging(self.embed_dim * 4)  # 384 -> 768


        self.bottleneck = nn.Sequential(
            SimplifiedSwinBlock(self.embed_dim * 8, num_heads=24),
            SimplifiedSwinBlock(self.embed_dim * 8, num_heads=24),
            SimplifiedSwinBlock(self.embed_dim * 8, num_heads=24),
        )


        self.agent_attn1 = SimpleAgentAttention(self.embed_dim)
        self.agent_attn2 = SimpleAgentAttention(self.embed_dim * 2)
        self.agent_attn3 = SimpleAgentAttention(self.embed_dim * 4)
        self.agent_attn4 = SimpleAgentAttention(self.embed_dim * 8)

        self.upsample1 = SimplePatchExpanding(self.embed_dim * 8)  # 768 -> 384
        self.upsample2 = SimplePatchExpanding(self.embed_dim * 4)  # 384 -> 192
        self.upsample3 = SimplePatchExpanding(self.embed_dim * 2)  # 192 -> 96


        self.decoder_stage1 = self._make_decoder_stage(self.embed_dim * 4, 2, 12)  # 384
        self.decoder_stage2 = self._make_decoder_stage(self.embed_dim * 2, 6, 6)   # 192
        self.decoder_stage3 = self._make_decoder_stage(self.embed_dim, 2, 3)       # 96
        self.decoder_stage4 = self._make_decoder_stage(self.embed_dim, 2, 3)       # 96


        # Fusion 1: (384 + 384) -> 384
        self.fusion1 = nn.Conv2d(self.embed_dim * 8, self.embed_dim * 4, kernel_size=1)
        # Fusion 2: (192 + 192) -> 192
        self.fusion2 = nn.Conv2d(self.embed_dim * 4, self.embed_dim * 2, kernel_size=1)
        # Fusion 3: (96 + 96) -> 96
        self.fusion3 = nn.Conv2d(self.embed_dim * 2, self.embed_dim, kernel_size=1)


        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.final_proj = nn.Linear(self.embed_dim, num_classes)

    def _make_encoder_stage(self, dim, num_blocks, num_heads):
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                SimplifiedSwinBlock(dim, num_heads=num_heads)
            )
        return nn.Sequential(*blocks)

    def _make_decoder_stage(self, dim, num_blocks, num_heads):
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                SimplifiedSwinBlock(dim, num_heads=num_heads)
            )
        return nn.Sequential(*blocks)

    def forward(self, x):
        B, C, H_img, W_img = x.shape


        x = self.patch_embed(x)  # [B, embed_dim, H, W]
        B, C_emb, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]


        x = x + self.pos_embed


        x_init = x
        H_init, W_init = H, W


        x = self.encoder_stage1[0](x, H, W)
        skip1 = x


        x, H, W = self.downsample1(x)


        x = self.encoder_stage2[0](x, H, W)
        skip2 = x


        x, H, W = self.downsample2(x)


        x = self.encoder_stage3[0](x, H, W)
        skip3 = x


        x, H, W = self.downsample3(x)


        x = self.encoder_stage4[0](x, H, W)


        for block in self.bottleneck:
            x = block(x, H, W)


        skip1_attn = self.apply_agent_attention(skip1, self.agent_attn1, H_init, W_init)
        skip2_attn = self.apply_agent_attention(skip2, self.agent_attn2, H_init//2, W_init//2)
        skip3_attn = self.apply_agent_attention(skip3, self.agent_attn3, H_init//4, W_init//4)
        x_attn = self.apply_agent_attention(x, self.agent_attn4, H_init//8, W_init//8)


        x, H, W = self.upsample1(x_attn, H_init//8, W_init//8) # 768 -> 384


        x = torch.cat([x, skip3_attn], dim=-1)

        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.fusion1(x)
        x = x.flatten(2).transpose(1, 2) # [B, HW, 384]


        x = self.decoder_stage1[0](x, H, W)

        x, H, W = self.upsample2(x, H_init//4, W_init//4) # 384 -> 192


        x = torch.cat([x, skip2_attn], dim=-1)

        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.fusion2(x)
        x = x.flatten(2).transpose(1, 2) # [B, HW, 192]

        x = self.decoder_stage2[0](x, H, W)


        x, H, W = self.upsample3(x, H_init//2, W_init//2) # 192 -> 96


        x = torch.cat([x, skip1_attn], dim=-1)

        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.fusion3(x)
        x = x.flatten(2).transpose(1, 2) # [B, HW, 96]


        x = self.decoder_stage3[0](x, H, W)


        x = self.decoder_stage4[0](x, H, W)


        x = self.final_norm(x)
        x = self.final_proj(x)


        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = nn.functional.interpolate(x, size=(H_img, W_img), mode='bilinear', align_corners=False)

        return x

    def apply_agent_attention(self, x, agent_attn, H, W):

        B, L, C = x.shape


        x_2d = x.transpose(1, 2).reshape(B, C, H, W)


        x_attn = agent_attn(x_2d)


        x_attn = x_attn.flatten(2).transpose(1, 2)

        return x_attn

class CompoundLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, num_classes=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes

    def dice_loss(self, pred, target):
        smooth = 1e-5
        pred = torch.softmax(pred, dim=1)
        loss = 0
        for cls in range(self.num_classes):
            pred_cls = pred[:, cls]
            target_cls = (target == cls).float()
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            loss += 1 - (2 * intersection + smooth) / (union + smooth)
        return loss / self.num_classes

    def forward(self, pred, target):
        ce_loss = nn.functional.cross_entropy(pred, target)
        dice = self.dice_loss(pred, target)

        total_loss = self.alpha * ce_loss + self.beta * dice
        return total_loss

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    torch.cuda.empty_cache()

    for images, labels in tqdm(loader, desc="训练"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device, num_classes=2):
    model.eval()
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="评估"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            preds_np = preds.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()


            batch_conf_mat = confusion_matrix(
                labels_np, preds_np,
                labels=range(num_classes)
            )
            conf_mat += batch_conf_mat


    PA_per_class = []
    IoU_per_class = []

    for i in range(num_classes):
        TP = conf_mat[i, i]
        total_pred = conf_mat[:, i].sum()
        total_true = conf_mat[i, :].sum()

        PA = TP / total_true if total_true > 0 else 0
        IoU = TP / (total_true + total_pred - TP) if (total_true + total_pred - TP) > 0 else 0

        PA_per_class.append(PA)
        IoU_per_class.append(IoU)

    mPA = np.mean(PA_per_class)
    mIoU = np.mean(IoU_per_class)

    return mPA, mIoU, PA_per_class, IoU_per_class, conf_mat


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model = SimpleAgentSwinUNet(num_classes=2, img_size=224).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

criterion = CompoundLoss(alpha=1.0, beta=1.0, gamma=0.5)

num_epochs = 50
best_mIoU = 0
best_model_path = '/content/drive/MyDrive/best_agent_swin_unet.pth'

history = {
    'train_loss': [],
    'val_mPA': [],
    'val_mIoU': []
}

print("开始训练Agent Swin-UNet...")
for epoch in raAgent Swin-UNete(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    # 训练阶段
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    history['train_loss'].append(train_loss)

    # 验证阶段
    mPA, mIoU, PA_class, IoU_class, _ = evaluate(model, val_loader, device)
    history['val_mPA'].append(mPA)
    history['val_mIoU'].append(mIoU)

    print(f"训练损失: {train_loss:.4f}")
    print(f"验证集 - mPA: {mPA:.4f}, mIoU: {mIoU:.4f}")

    for i, (pa, iou) in enumerate(zip(PA_class, IoU_class)):
        print(f"  类别 {i}: PA={pa:.4f}, IoU={iou:.4f}")

