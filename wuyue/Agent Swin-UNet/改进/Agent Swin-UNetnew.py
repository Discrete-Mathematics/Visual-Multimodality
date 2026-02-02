
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
import random
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

drive.mount('/content/drive')

DATASET_PATH = '/content/drive/MyDrive/MUSeg'
TRAIN_RGB_DIR = os.path.join(DATASET_PATH, 'RGB')
TRAIN_LABEL_DIR = os.path.join(DATASET_PATH, 'Label')
for split_file in ['train.txt', 'val.txt']:
    file_path = os.path.join(DATASET_PATH, split_file)
    if os.path.exists(file_path):
        print(f"找到: {file_path}")
        with open(file_path, 'r') as f:
            lines = f.readlines()
            print(f"  包含 {len(lines)} 个样本")
    else:

SPLIT_FILES = {
    'train': os.path.join(DATASET_PATH, 'train.txt'),
    'val': os.path.join(DATASET_PATH, 'val.txt')
}

class MUSegMultiClassDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, image_size=(224, 224)):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.num_classes = 16 

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

        label = torch.clamp(label, min=0, max=15)

        return image, label

image_size = (224, 224)
train_transform = A.Compose([
    A.Resize(*image_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
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

# 创建数据集
try:
    train_dataset = MUSegMultiClassDataset(DATASET_PATH, split='train', transform=train_transform)
    val_dataset = MUSegMultiClassDataset(DATASET_PATH, split='val', transform=val_transform)
    test_dataset = val_dataset  # 使用验证集作为测试集

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = val_loader

    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")
    print(f"测试集: 使用验证集 ({len(val_dataset)} 张)")

    print("\n正在验证数据标签范围 (前50个样本)...")
    valid_data = True
    max_val = 0
    min_val = 255
    for i in range(min(len(train_dataset), 50)):
        _, lbl = train_dataset[i]
        max_val = max(max_val, lbl.max().item())
        min_val = min(min_val, lbl.min().item())

    print(f"标签值范围: min={min_val}, max={max_val}")
    if max_val <= 15 and min_val >= 0:
        print("✓ 数据验证通过：标签值在0-15范围内。")
    else:
        raise ValueError("数据集标签存在异常值，请检查数据处理逻辑！")

except Exception as e:
    print(f"数据集加载错误: {e}")
    raise

class SimpleAgentAttention(nn.Module):
    """简化的代理注意力模块"""
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
        agent_weights = self.softmax(agent_logits)

        value = self.value_conv(x).view(B, C, N)
        aggregated_agents = torch.bmm(agent_weights, value.transpose(1, 2))

        query = self.query_conv(x).view(B, C, N)
        broadcast_scores = torch.bmm(query.transpose(1, 2), aggregated_agents.transpose(1, 2)) / (C ** 0.5)
        broadcast_weights = self.softmax(broadcast_scores)

        broadcast_features = torch.bmm(broadcast_weights, aggregated_agents)
        broadcast_features = broadcast_features.transpose(1, 2).view(B, C, H, W)

        local_features = self.depthwise_conv(x)
        local_features = self.pointwise_conv(local_features)

        combined = broadcast_features + local_features
        output = self.output_proj(combined)
        output = output + x

        return output

class SimplifiedSwinBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True
        )

        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, in_channels),
            nn.Dropout(drop)
        )

        self.dropout = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x

        x_norm = self.norm1(x)
        x_attn, _ = self.attention(x_norm, x_norm, x_norm)
        x = shortcut + self.dropout(x_attn)

        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = x + self.dropout(x_mlp)

        return x


class SimplePatchMerging(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2
        self.conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(self.out_channels)

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = H

        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        x_down = self.conv(x_2d)

        B, C_new, H_new, W_new = x_down.shape
        x_down = x_down.flatten(2).transpose(1, 2)
        x_down = self.norm(x_down)

        return x_down, H_new, W_new

class SimplePatchExpanding(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // 2
        self.conv = nn.ConvTranspose2d(in_channels, self.out_channels, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(self.out_channels)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        x_up = self.conv(x_2d)

        B, C_new, H_new, W_new = x_up.shape
        x_up = x_up.flatten(2).transpose(1, 2)
        x_up = self.norm(x_up)

        return x_up, H_new, W_new

class SimpleAgentSwinUNetMultiClass(nn.Module):
    def __init__(self, num_classes=16, img_size=224):
        super().__init__()
        self.img_size = img_size
        self.patch_size = 4
        self.embed_dim = 96

        self.patch_embed = nn.Conv2d(
            3, self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, (img_size // self.patch_size) ** 2, self.embed_dim)
        )

        # 编码器阶段
        self.encoder_stage1 = self._make_encoder_stage(self.embed_dim, 2, 3)
        self.encoder_stage2 = self._make_encoder_stage(self.embed_dim * 2, 2, 6)
        self.encoder_stage3 = self._make_encoder_stage(self.embed_dim * 4, 6, 12)
        self.encoder_stage4 = self._make_encoder_stage(self.embed_dim * 8, 2, 24)

        # 下采样层
        self.downsample1 = SimplePatchMerging(self.embed_dim)
        self.downsample2 = SimplePatchMerging(self.embed_dim * 2)
        self.downsample3 = SimplePatchMerging(self.embed_dim * 4)

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            SimplifiedSwinBlock(self.embed_dim * 8, num_heads=24),
            SimplifiedSwinBlock(self.embed_dim * 8, num_heads=24),
            SimplifiedSwinBlock(self.embed_dim * 8, num_heads=24),
        )

        # 跳跃连接中的代理注意力
        self.agent_attn1 = SimpleAgentAttention(self.embed_dim)
        self.agent_attn2 = SimpleAgentAttention(self.embed_dim * 2)
        self.agent_attn3 = SimpleAgentAttention(self.embed_dim * 4)
        self.agent_attn4 = SimpleAgentAttention(self.embed_dim * 8)

        # 上采样层
        self.upsample1 = SimplePatchExpanding(self.embed_dim * 8)
        self.upsample2 = SimplePatchExpanding(self.embed_dim * 4)
        self.upsample3 = SimplePatchExpanding(self.embed_dim * 2)

        # 解码器阶段
        self.decoder_stage1 = self._make_decoder_stage(self.embed_dim * 4, 2, 12)
        self.decoder_stage2 = self._make_decoder_stage(self.embed_dim * 2, 6, 6)
        self.decoder_stage3 = self._make_decoder_stage(self.embed_dim, 2, 3)
        self.decoder_stage4 = self._make_decoder_stage(self.embed_dim, 2, 3)

        # 融合层
        self.fusion1 = nn.Conv2d(self.embed_dim * 8, self.embed_dim * 4, kernel_size=1)
        self.fusion2 = nn.Conv2d(self.embed_dim * 4, self.embed_dim * 2, kernel_size=1)
        self.fusion3 = nn.Conv2d(self.embed_dim * 2, self.embed_dim, kernel_size=1)

        # 最终输出层
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.final_proj = nn.Linear(self.embed_dim, num_classes)

    def _make_encoder_stage(self, dim, num_blocks, num_heads):
        blocks = []
        for i in range(num_blocks):
            blocks.append(SimplifiedSwinBlock(dim, num_heads=num_heads))
        return nn.Sequential(*blocks)

    def _make_decoder_stage(self, dim, num_blocks, num_heads):
        blocks = []
        for i in range(num_blocks):
            blocks.append(SimplifiedSwinBlock(dim, num_heads=num_heads))
        return nn.Sequential(*blocks)

    def forward(self, x):
        B, C, H_img, W_img = x.shape

        # 补丁嵌入
        x = self.patch_embed(x)
        B, C_emb, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        # 保存初始特征
        x_init = x
        H_init, W_init = H, W

        # 编码器阶段1
        x = self.encoder_stage1[0](x, H, W)
        skip1 = x

        # 下采样1
        x, H, W = self.downsample1(x)

        # 编码器阶段2
        x = self.encoder_stage2[0](x, H, W)
        skip2 = x

        # 下采样2
        x, H, W = self.downsample2(x)

        # 编码器阶段3
        x = self.encoder_stage3[0](x, H, W)
        skip3 = x

        # 下采样3
        x, H, W = self.downsample3(x)

        # 编码器阶段4
        x = self.encoder_stage4[0](x, H, W)

        # 瓶颈层
        for block in self.bottleneck:
            x = block(x, H, W)

        # 对跳跃连接特征应用代理注意力
        skip1_attn = self.apply_agent_attention(skip1, self.agent_attn1, H_init, W_init)
        skip2_attn = self.apply_agent_attention(skip2, self.agent_attn2, H_init//2, W_init//2)
        skip3_attn = self.apply_agent_attention(skip3, self.agent_attn3, H_init//4, W_init//4)
        x_attn = self.apply_agent_attention(x, self.agent_attn4, H_init//8, W_init//8)

        # 上采样1 + 跳跃连接
        x, H, W = self.upsample1(x_attn, H_init//8, W_init//8)
        x = torch.cat([x, skip3_attn], dim=-1)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.fusion1(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.decoder_stage1[0](x, H, W)

        # 上采样2 + 跳跃连接
        x, H, W = self.upsample2(x, H_init//4, W_init//4)
        x = torch.cat([x, skip2_attn], dim=-1)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.fusion2(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.decoder_stage2[0](x, H, W)

        # 上采样3 + 跳跃连接
        x, H, W = self.upsample3(x, H_init//2, W_init//2)
        x = torch.cat([x, skip1_attn], dim=-1)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.fusion3(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.decoder_stage3[0](x, H, W)

        # 最终解码器阶段
        x = self.decoder_stage4[0](x, H, W)

        # 最终投影
        x = self.final_norm(x)
        x = self.final_proj(x)

        # 重塑为2D并插值到原始大小
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = nn.functional.interpolate(x, size=(H_img, W_img), mode='bilinear', align_corners=False)

        return x

    def apply_agent_attention(self, x, agent_attn, H, W):
        B, L, C = x.shape
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        x_attn = agent_attn(x_2d)
        x_attn = x_attn.flatten(2).transpose(1, 2)
        return x_attn

class CompoundLossMultiClass(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, num_classes=16, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes

        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights).float()
        else:
            self.class_weights = None

    def dice_loss(self, pred, target):
        smooth = 1e-5
        pred = torch.softmax(pred, dim=1)
        loss = 0

        for cls in range(self.num_classes):
            pred_cls = pred[:, cls]
            target_cls = (target == cls).float()

            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()

            if self.class_weights is not None:
                loss += self.class_weights[cls] * (1 - (2 * intersection + smooth) / (union + smooth))
            else:
                loss += 1 - (2 * intersection + smooth) / (union + smooth)

        return loss / self.num_classes

    def forward(self, pred, target):
        if self.class_weights is not None:
            ce_loss = nn.functional.cross_entropy(pred, target, weight=self.class_weights.to(pred.device))
        else:
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

def evaluate(model, loader, device, num_classes=16):
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

            # 确保混淆矩阵形状正确
            if batch_conf_mat.shape == (num_classes, num_classes):
                conf_mat += batch_conf_mat
            else:
                # 处理形状不匹配的情况
                temp_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
                min_rows = min(batch_conf_mat.shape[0], num_classes)
                min_cols = min(batch_conf_mat.shape[1], num_classes)
                temp_mat[:min_rows, :min_cols] = batch_conf_mat[:min_rows, :min_cols]
                conf_mat += temp_mat

    # 计算各类指标
    PA_per_class = []
    IoU_per_class = []

    for i in range(num_classes):
        TP = conf_mat[i, i]
        total_pred = conf_mat[:, i].sum()
        total_true = conf_mat[i, :].sum()

        if total_true > 0:
            PA = TP / total_true
        else:
            PA = 0

        if (total_true + total_pred - TP) > 0:
            IoU = TP / (total_true + total_pred - TP)
        else:
            IoU = 0

        PA_per_class.append(PA)
        IoU_per_class.append(IoU)

    # 计算全局指标
    total_TP = np.trace(conf_mat)
    total_pixels = conf_mat.sum()

    if total_pixels > 0:
        mPA = total_TP / total_pixels
    else:
        mPA = 0

    mIoU = np.mean([iou for iou in IoU_per_class if not np.isnan(iou)])

    return mPA, mIoU, PA_per_class, IoU_per_class, conf_mat


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 实验参数
num_experiments = 100
num_epochs_per_experiment = 100
num_classes = 16

# 类别名称映射
class_names = [
    'background', 'person', 'cable', 'tube', 'indicator',
    'electrical equipment', 'electronic equipment', 'mining equipment',
    'rail area', 'support equipment', 'door', 'tools and materials',
    'rescue equipment', 'container', 'metal fixture', 'anchoring equipment'
]

# 创建结果保存目录
results_dir = '/content/drive/MyDrive/MUSeg_Experiments'
os.makedirs(results_dir, exist_ok=True)

# 初始化结果存储
all_results = {
    'experiment_id': [],
    'best_mPA': [],
    'best_mIoU': [],
    'best_epoch': [],
    'total_training_time': []
}

# 为每个类别添加存储
for i, class_name in enumerate(class_names):
    all_results[f'class_{i}_best_PA'] = []
    all_results[f'class_{i}_best_IoU'] = []

print(f"开始进行 {num_experiments} 次实验，每次 {num_epochs_per_experiment} 个epoch")
print("="*80)

for exp_id in range(num_experiments):
    print(f"\n{'='*60}")
    print(f"实验 #{exp_id+1}/{num_experiments}")
    print(f"{'='*60}")

    # 设置随机种子以确保实验可重复性
    seed = 42 + exp_id
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 创建模型
    model = SimpleAgentSwinUNetMultiClass(num_classes=num_classes, img_size=224).to(device)

    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_per_experiment)

    # 损失函数
    criterion = CompoundLossMultiClass(alpha=1.0, beta=1.0, gamma=0.5, num_classes=num_classes)

    # 训练参数
    best_mIoU = 0
    best_mPA = 0
    best_epoch = 0
    best_class_PA = [0] * num_classes
    best_class_IoU = [0] * num_classes

    # 实验开始时间
    start_time = time.time()

    print(f"随机种子: {seed}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 单个实验的训练循环
    experiment_history = {
        'train_loss': [],
        'val_mPA': [],
        'val_mIoU': []
    }

    for epoch in range(num_epochs_per_experiment):
        print(f"\nEpoch {epoch+1}/{num_epochs_per_experiment}")

        # 训练阶段
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        experiment_history['train_loss'].append(train_loss)

        # 验证阶段
        mPA, mIoU, PA_class, IoU_class, _ = evaluate(model, val_loader, device, num_classes=num_classes)
        experiment_history['val_mPA'].append(mPA)
        experiment_history['val_mIoU'].append(mIoU)

        print(f"训练损失: {train_loss:.4f}")
        print(f"验证集 - mPA: {mPA:.4f}, mIoU: {mIoU:.4f}")

        # 保存最佳结果
        if mIoU > best_mIoU:
            best_mIoU = mIoU
            best_mPA = mPA
            best_epoch = epoch + 1
            best_class_PA = PA_class.copy()
            best_class_IoU = IoU_class.copy()

            # 保存最佳模型
            model_path = os.path.join(results_dir, f'experiment_{exp_id+1}_best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mIoU': best_mIoU,
                'best_mPA': best_mPA,
                'experiment_history': experiment_history
            }, model_path)

            print(f"✓ 保存最佳模型 (mIoU: {best_mIoU:.4f}, mPA: {best_mPA:.4f})")

        # 更新学习率
        scheduler.step()

    # 实验结束时间
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n实验 #{exp_id+1} 完成")
    print(f"总训练时间: {total_time:.2f} 秒")
    print(f"最佳结果在第 {best_epoch} 个epoch")
    print(f"最佳 mPA: {best_mPA:.4f}")
    print(f"最佳 mIoU: {best_mIoU:.4f}")

    # 存储实验结果
    all_results['experiment_id'].append(exp_id+1)
    all_results['best_mPA'].append(best_mPA)
    all_results['best_mIoU'].append(best_mIoU)
    all_results['best_epoch'].append(best_epoch)
    all_results['total_training_time'].append(total_time)

    for i in range(num_classes):
        all_results[f'class_{i}_best_PA'].append(best_class_PA[i])
        all_results[f'class_{i}_best_IoU'].append(best_class_IoU[i])

    # 打印每个类别的最佳结果
    print("\n每个类别的最佳结果:")
    print("-" * 80)
    print(f"{'类别':<25} {'PA':<10} {'IoU':<10}")
    print("-" * 80)
    for i in range(num_classes):
        print(f"{class_names[i]:<25} {best_class_PA[i]:<10.4f} {best_class_IoU[i]:<10.4f}")

    # 保存当前实验的详细结果
    exp_result = {
        'experiment_id': exp_id + 1,
        'seed': seed,
        'best_epoch': best_epoch,
        'best_mPA': best_mPA,
        'best_mIoU': best_mIoU,
        'total_training_time': total_time,
        'class_results': {}
    }

    for i in range(num_classes):
        exp_result['class_results'][class_names[i]] = {
            'PA': float(best_class_PA[i]),
            'IoU': float(best_class_IoU[i])
        }

    exp_result_path = os.path.join(results_dir, f'experiment_{exp_id+1}_results.json')
    with open(exp_result_path, 'w') as f:
        json.dump(exp_result, f, indent=4)

    print(f"实验结果已保存到: {exp_result_path}")

results_df = pd.DataFrame(all_results)

# 找到最佳实验
best_exp_idx = results_df['best_mIoU'].idxmax()
best_exp_id = results_df.loc[best_exp_idx, 'experiment_id']
best_mIoU_value = results_df.loc[best_exp_idx, 'best_mIoU']
best_mPA_value = results_df.loc[best_exp_idx, 'best_mPA']

print(f"\n最佳实验: #{best_exp_id}")
print(f"最佳 mIoU: {best_mIoU_value:.4f}")
print(f"最佳 mPA: {best_mPA_value:.4f}")

# 计算每个类别的平均指标
print("\n每个类别的平均指标 (所有实验):")
print("-" * 80)
print(f"{'类别':<25} {'平均PA':<10} {'平均IoU':<10} {'PA标准差':<10} {'IoU标准差':<10}")
print("-" * 80)

class_avg_results = []
for i, class_name in enumerate(class_names):
    pa_col = f'class_{i}_best_PA'
    iou_col = f'class_{i}_best_IoU'

    avg_pa = results_df[pa_col].mean()
    std_pa = results_df[pa_col].std()
    avg_iou = results_df[iou_col].mean()
    std_iou = results_df[iou_col].std()

    class_avg_results.append({
        'class_name': class_name,
        'avg_PA': avg_pa,
        'std_PA': std_pa,
        'avg_IoU': avg_iou,
        'std_IoU': std_iou
    })

    print(f"{class_name:<25} {avg_pa:<10.4f} {avg_iou:<10.4f} {std_pa:<10.4f} {std_iou:<10.4f}")



plt.subplot(2, 3, 5)
class_pas = [result['avg_PA'] for result in class_avg_results]
plt.bar(range(len(class_names)), class_pas, color='lightcoral', edgecolor='black')
plt.xticks(range(len(class_names)), [name[:10] for name in class_names], rotation=45, ha='right')
plt.xlabel('类别')
plt.ylabel('平均PA')
plt.title('每个类别的平均PA')
plt.tight_layout()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 6)
plt.scatter(results_df['best_mPA'], results_df['best_mIoU'], alpha=0.6, edgecolors='black')
plt.xlabel('mPA')
plt.ylabel('mIoU')
plt.title('mPA vs mIoU 关系')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'experiment_statistics.png'), dpi=300, bbox_inches='tight')
plt.show()
