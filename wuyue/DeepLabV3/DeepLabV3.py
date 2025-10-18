!pip install albumentations opencv-python scikit-image

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import time

# 矿井环境图像增强
class MineImageEnhancement:
    """
    针对矿井环境的图像增强方法
    基于CLAHE和对比度增强技术
    """

    @staticmethod
    def adaptive_contrast_enhancement(image, clip_limit=2.0, tile_grid_size=(8,8)):
        if image is None:
            return image

        try:
            # 转换为YUV颜色空间
            if len(image.shape) == 3:
                yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                y, u, v = cv2.split(yuv)
            else:
                y = image

            # CLAHE对比度限制直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            y_enhanced = clahe.apply(y)

            if len(image.shape) == 3:
                yuv_enhanced = cv2.merge([y_enhanced, u, v])
                enhanced = cv2.cvtColor(yuv_enhanced, cv2.COLOR_YUV2RGB)
                return enhanced
            else:
                return y_enhanced
        except Exception as e:
            print(f"对比度增强失败: {e}")
            return image
# 平衡的数据预处理
IMAGE_SIZE = 512 

def get_train_transforms():
    return A.Compose([
        # 固定尺寸
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        # 对矿井环境重要的几何变换
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,  # 减少变换幅度
            scale_limit=0.05,
            rotate_limit=10,
            p=0.3,
            border_mode=0
        ),

        # 对矿井环境重要的颜色增强
        A.RandomBrightnessContrast(
            brightness_limit=0.2,  # 减少变化范围
            contrast_limit=0.2,
            p=0.5
        ),
        A.CLAHE(clip_limit=3.0, p=0.4),  # 保留CLAHE，对低光照重要

        # 标准化
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

def get_val_transforms():
    """验证时的数据变换"""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

# Dataset类
class BalancedMUSegDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None, is_train=True, apply_enhancement=True):
        self.root_dir = root_dir
        with open(txt_file, "r") as f:
            self.ids = [line.strip() for line in f.readlines()]
        self.transform = transform
        self.is_train = is_train
        self.apply_enhancement = apply_enhancement
        self.enhancement = MineImageEnhancement()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        try:
            img_id = self.ids[idx]
            img_path = os.path.join(self.root_dir, "RGB", f"{img_id}.jpg")
            mask_path = os.path.join(self.root_dir, "Label", f"{img_id}_label.png")

            # 加载图像和掩码
            image = cv2.imread(img_path)
            if image is None:
                print(f"无法加载图像: {img_path}")
                return self._create_dummy_data()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"无法加载掩码: {mask_path}")
                return self._create_dummy_data()

            # 应用矿井图像增强 - 适度应用
            if self.apply_enhancement and self.is_train and random.random() > 0.5:
                image = self.enhancement.adaptive_contrast_enhancement(image)

            # 应用Albumentations变换
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
                # 确保mask是long类型
                mask = mask.long()

            return image, mask

        except Exception as e:
            print(f"处理图像 {self.ids[idx]} 时出错: {e}")
            return self._create_dummy_data()

    def _create_dummy_data(self):
        dummy_image = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
        dummy_mask = torch.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=torch.long)
        return dummy_image, dummy_mask

def get_optimal_batch_size():
    if torch.cuda.is_available():
        # 获取GPU信息
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if total_memory >= 14: 
            return 6
        elif total_memory >= 10: 
            return 4
        elif total_memory >= 6: 
            return 3
        else:  
            return 2
    else:
        return 2  
def balanced_evaluate_model(model, loader, num_classes, device):
    model.eval()
    # 使用torch张量进行GPU加速计算
    confusion_matrix = torch.zeros(num_classes, num_classes,
                                  dtype=torch.int64, device=device)
    total_pixels = 0
    correct_pixels = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="验证进度"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # 计算像素准确率
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

            # 精确的混淆矩阵计算
            masks_flat = masks.view(-1)
            preds_flat = preds.view(-1)

            # 使用高效的批量更新
            for cls_true in range(num_classes):
                true_mask = (masks_flat == cls_true)
                for cls_pred in range(num_classes):
                    pred_mask = (preds_flat == cls_pred)
                    confusion_matrix[cls_true, cls_pred] += (true_mask & pred_mask).sum()

    # 计算各项指标
    PA = correct_pixels / total_pixels if total_pixels > 0 else 0

    # 计算各类IoU
    intersection = torch.diag(confusion_matrix)
    union = confusion_matrix.sum(1) + confusion_matrix.sum(0) - intersection
    iou_per_class = torch.zeros(num_classes, device=device)
    for i in range(num_classes):
        if union[i] > 0:
            iou_per_class[i] = intersection[i] / union[i]

    valid_iou = iou_per_class[~iou_per_class.isnan()]
    mIoU = valid_iou.mean().item() if len(valid_iou) > 0 else 0
    # 计算mPA
    pa_per_class = torch.zeros(num_classes, device=device)
    for i in range(num_classes):
        total_class = confusion_matrix[i, :].sum()
        if total_class > 0:
            pa_per_class[i] = confusion_matrix[i, i] / total_class

    valid_pa = pa_per_class[~pa_per_class.isnan()]
    mPA = valid_pa.mean().item() if len(valid_pa) > 0 else 0

    return PA, mPA, mIoU, iou_per_class.cpu().numpy()
# 模型定义

class MineDeepLabV3(nn.Module):

    def __init__(self, num_classes):
        super(MineDeepLabV3, self).__init__()
        self.model = deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']

def balanced_train_model(model, train_loader, val_loader, num_classes, epochs=100, save_path="/content/drive/MyDrive/mine_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_mIoU = 0.0
    train_losses = []
    val_metrics = []
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        start_time = time.time()
        pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}/{epochs}")
        for images, masks in pbar:
            try:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            except Exception as e:
                continue

        train_time = time.time() - start_time

        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            train_losses.append(avg_loss)
        else:
            avg_loss = 0
            train_losses.append(0)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == epochs:
            print("开始验证...")
            val_start_time = time.time()
            PA, mPA, mIoU, iou_details = balanced_evaluate_model(model, val_loader, num_classes, device)
            val_time = time.time() - val_start_time
            val_metrics.append((PA, mPA, mIoU, epoch))
            print(f"\nEpoch {epoch+1}/{epochs} 完成")
            print(f"训练时间: {train_time:.1f}s, 验证时间: {val_time:.1f}s")
            print(f"训练损失: {avg_loss:.4f}")
            print(f"当前学习率: {current_lr:.8f}")
            print(f"验证集指标 => PA: {PA:.4f}, mPA: {mPA:.4f}, mIoU: {mIoU:.4f}")
            # 保存最佳模型
            if mIoU > best_mIoU:
                best_mIoU = mIoU
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_mIoU': best_mIoU,
                    'val_metrics': val_metrics,
                    'train_losses': train_losses
                }, save_path)
                print(f" 保存最佳模型 (mIoU={best_mIoU:.4f})")
            # 每10个epoch显示一次各类别IoU
            if (epoch + 1) % 10 == 0:
                print("各类别IoU详情:")
                for i, iou in enumerate(iou_details):
                    if iou > 0.01:  # 只显示有意义的类别
                        print(f"  类别 {i}: {iou:.4f}")

            print("-" * 60)
        else:
            print(f"\nEpoch {epoch+1}/{epochs} 完成 - 训练损失: {avg_loss:.4f}, 学习率: {current_lr:.8f}")
            print("-" * 60)

    return train_losses, val_metrics

# 可视化函数
def balanced_visualize_results(model, dataset, num_samples=5, save_dir="/content/drive/MyDrive/visualizations"):

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    indices = list(range(min(num_samples, len(dataset))))

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            image, true_mask = dataset[sample_idx]

            # 推理
            pred = model(image.unsqueeze(0).to(device))
            pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

            # 反标准化图像用于显示
            image_np = image.cpu().numpy()
            image_np = np.transpose(image_np, (1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = std * image_np + mean
            image_np = np.clip(image_np, 0, 1)

            # 应用增强用于对比
            enhanced_image = MineImageEnhancement.adaptive_contrast_enhancement((image_np * 255).astype(np.uint8))
            if enhanced_image is not None:
                enhanced_image = enhanced_image.astype(np.float32) / 255.0
            else:
                enhanced_image = image_np

            # 绘制结果
            axes[idx, 0].imshow(image_np)
            axes[idx, 0].set_title("原始图像")
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(enhanced_image)
            axes[idx, 1].set_title("增强图像")
            axes[idx, 1].axis('off')

            axes[idx, 2].imshow(true_mask.numpy(), cmap='tab20', vmin=0, vmax=15)
            axes[idx, 2].set_title("真值标签")
            axes[idx, 2].axis('off')

            axes[idx, 3].imshow(pred_mask, cmap='tab20', vmin=0, vmax=15)
            axes[idx, 3].set_title("预测结果")
            axes[idx, 3].axis('off')

            # 保存单独的图像
            plt.figure(figsize=(12, 3))
            plt.subplot(1, 4, 1)
            plt.imshow(image_np)
            plt.title("原始图像")
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.imshow(enhanced_image)
            plt.title("增强图像")
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.imshow(true_mask.numpy(), cmap='tab20', vmin=0, vmax=15)
            plt.title("真值标签")
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.imshow(pred_mask, cmap='tab20', vmin=0, vmax=15)
            plt.title("预测结果")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(f"{save_dir}/sample_{idx}.png", dpi=150, bbox_inches='tight')
            plt.close()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/all_samples.png", dpi=150, bbox_inches='tight')
    plt.show()
def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 数据路径
    data_root = "/content/drive/MyDrive/MUSeg"
    train_txt = os.path.join(data_root, "train.txt")
    val_txt = os.path.join(data_root, "val.txt")
    # 检查数据路径
    if not os.path.exists(data_root):
        return
    # 创建数据集
    train_dataset = BalancedMUSegDataset(
        data_root, train_txt,
        transform=get_train_transforms(),
        is_train=True,
        apply_enhancement=True
    )

    val_dataset = BalancedMUSegDataset(
        data_root, val_txt,
        transform=get_val_transforms(),
        is_train=False,
        apply_enhancement=False
    )

    # 动态调整批次大小
    batch_size = get_optimal_batch_size()
    print(f" 自动选择批次大小: {batch_size}")

    # 数据加载器 - 平衡设置
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # 平衡的worker数量
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # 平衡的worker数量
        pin_memory=True,
        drop_last=False
    )

    print(f" 训练样本数: {len(train_dataset)}")
    print(f" 验证样本数: {len(val_dataset)}")

    # 测试一个批次确保尺寸正确
    try:
        test_images, test_masks = next(iter(train_loader))
        print(f" 批次图像尺寸: {test_images.shape}")
        print(f" 批次掩码尺寸: {test_masks.shape}")
    except Exception as e:
        print(f" 数据加载测试失败: {e}")
        return

    # 模型初始化
    num_classes = 16  # 15类 + 背景
    model = MineDeepLabV3(num_classes=num_classes)
    model = model.to(device)

    print(" 开始训练...")
    train_losses, val_metrics = balanced_train_model(
        model, train_loader, val_loader, num_classes,
        epochs=100,  
        save_path="/content/drive/MyDrive/mine_model_100epochs.pth"
    )

    # 加载最佳模型进行可视化
    try:
        checkpoint = torch.load("/content/drive/MyDrive/mine_model_100epochs.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f" 加载最佳模型，mIoU: {checkpoint['best_mIoU']:.4f}")
    except Exception as e:
        print(f" 加载最佳模型失败: {e}")
        print("使用当前模型进行可视化")

    # 生成5张可视化结果
    print(" 生成5张可视化结果...")
    balanced_visualize_results(model, val_dataset, num_samples=5)

    # 最终评估
    print(" 进行最终评估...")
    PA, mPA, mIoU, iou_details = balanced_evaluate_model(model, val_loader, num_classes, device)

    print("\n" + "="*60)
    print(" 最终评估结果 (100 Epochs)")
    print("="*60)
    print(f" 像素准确率 (PA): {PA:.4f} ({PA*100:.2f}%)")
    print(f" 平均像素准确率 (mPA): {mPA:.4f} ({mPA*100:.2f}%)")
    print(f" 平均交并比 (mIoU): {mIoU:.4f} ({mIoU*100:.2f}%)")
    
    # 显示最佳指标
    if val_metrics:
        best_pa = max([m[0] for m in val_metrics])
        best_mpa = max([m[1] for m in val_metrics])
        best_miou = max([m[2] for m in val_metrics])
        print(f" 训练最佳PA: {best_pa:.4f} ({best_pa*100:.2f}%)")
        print(f" 训练最佳mPA: {best_mpa:.4f} ({best_mpa*100:.2f}%)")
        print(f" 训练最佳mIoU: {best_miou:.4f} ({best_miou*100:.2f}%)")
    
    print("\n 各类别IoU详情:")
    for i, iou in enumerate(iou_details):
        if iou > 0.01: 
            print(f"   类别 {i:2d}: {iou:.4f} ({iou*100:.1f}%)")
    print("="*60)

# 启动主程序
if __name__ == "__main__":
    main()
