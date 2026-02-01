import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from sklearn.model_selection import train_test_split

# ====================== 配置参数 ======================
class Config:
    # 数据路径 - 模态1 (RGB图像)
    MODALITY1_DIR = Path("/root/autodl-tmp/dataset/Image")
    MODALITY1_EXT = ".jpg"
    
    # 数据路径 - 模态2 (深度图)
    MODALITY2_DIR = Path("/root/autodl-tmp/dataset/Depth")
    MODALITY2_EXT = ".png"
    
    # 标签路径
    MASK_DIR = Path("/root/autodl-tmp/dataset/Label_Label")
    MASK_EXT = ".png"
    
    # 模型输入尺寸
    CROP_SIZES = [512, 768]  # 多尺寸训练
    TARGET_SIZE = 256  # 最终统一尺寸
    
    # 增强参数
    RANDOM_CROP_PROB = 0.4
    RANDOM_SCALE_PROB = 0.3
    RESIZE_PROB = 0.3
    COLOR_AUG_PROB = 0.5  # 仅适用于RGB模态
    
    # 可视化
    VISUALIZE_SAMPLES = 5  # 可视化样本数量
    FIG_SIZE = (18, 16)

cfg = Config()

# ====================== 多模态数据增强管道 ======================
def create_multimodal_augmentation(target_size=512, apply_color_aug=True):
    """创建多模态数据增强管道"""
    # 基础空间变换 (适用于所有模态)
    spatial_transforms = [
        # 多模式裁剪策略
        A.OneOf([
            # 模式1：随机比例裁剪（保持物体完整性）
            A.RandomResizedCrop(
                size=(target_size, target_size),
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33),
                interpolation=cv2.INTER_LINEAR,
                p=cfg.RANDOM_SCALE_PROB
            ),
            
            # 模式2：固定尺寸随机位置裁剪
            A.RandomCrop(height=target_size, width=target_size, p=cfg.RANDOM_CROP_PROB),
            
            # 模式3：完整图像缩放
            A.Resize(height=target_size, width=target_size, p=cfg.RESIZE_PROB)
        ], p=1.0),
        
        # 空间增强
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.5),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-25, 25),
            shear=0,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            value=0,          # 原 cval
            mask_value=0,     # 原 cval_mask
            border_mode=cv2.BORDER_REFLECT101,  # 原 mode
            fit_output=False,
            p=0.5
        ),
    ]
    
    # 颜色增强 (仅适用于RGB模态)
    color_transforms = []
    if apply_color_aug:
        color_transforms = [
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
                A.RandomGamma(gamma_limit=(70, 140), p=1),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
            ], p=cfg.COLOR_AUG_PROB),
            
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
            
            # 传感器噪声
            A.OneOf([
                A.GaussNoise(var_limit=100.0, mean=0, p=1),
                A.GaussianBlur(blur_limit=(3, 7), p=1),
                A.MotionBlur(blur_limit=(5, 13), p=1),  # 确保奇数限制
            ], p=0.2),
            
            # 图像质量退化
            A.ImageCompression(quality_lower=75, quality_upper=95, p=0.2),
        ]
    
    # 通用变换 (适用于所有模态)
    common_transforms = [
        A.CoarseDropout(
            max_holes=12,
            max_height=target_size//10,
            max_width=target_size//10,
            min_holes=1,
            min_height=target_size//30,
            min_width=target_size//30,
            fill_value=0,
            p=0.2
        ),
        
        # 标准化和转换为张量
        ToTensorV2()
    ]
    
    # 组合变换
    all_transforms = spatial_transforms + color_transforms + common_transforms
    
    return A.Compose(all_transforms)
    
# ====================== 多模态语义分割数据集类 ======================
class MultimodalSegmentationDataset(Dataset):
    def __init__(self, modality1_paths, modality2_paths, mask_paths, 
                 modality1_transform=None, modality2_transform=None, mask_transform=None):
        self.modality1_paths = modality1_paths
        self.modality2_paths = modality2_paths
        self.mask_paths = mask_paths
        
        # 变换管道
        self.modality1_transform = modality1_transform
        self.modality2_transform = modality2_transform
        self.mask_transform = mask_transform
        
        # 验证数据配对
        self._validate_data()
        
    def _validate_data(self):
        """验证所有模态和mask的尺寸匹配"""
        for i, (m1_path, m2_path, mask_path) in enumerate(zip(
            self.modality1_paths, self.modality2_paths, self.mask_paths)):
            
            # 读取所有数据但不应用变换
            m1 = cv2.imread(str(m1_path))
            m2 = cv2.imread(str(m2_path), cv2.IMREAD_UNCHANGED)  # 保持原始格式
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if m1 is None:
                raise ValueError(f"无法读取模态1: {m1_path}")
            if m2 is None:
                raise ValueError(f"无法读取模态2: {m2_path}")
            if mask is None:
                raise ValueError(f"无法读取Mask: {mask_path}")
                
            # 检查尺寸匹配
            if m1.shape[:2] != mask.shape[:2]:
                raise ValueError(f"模态1和Mask尺寸不匹配: {m1.shape[:2]} vs {mask.shape[:2]} {m1_path}")
            if m2.shape[:2] != mask.shape[:2]:
                raise ValueError(f"模态2和Mask尺寸不匹配: {m2.shape[:2]} vs {mask.shape[:2]} {m2_path}")

    def __len__(self):
        return len(self.modality1_paths)
        
    def __getitem__(self, idx):
        # 读取模态1数据 (RGB)
        modality1_path = self.modality1_paths[idx]
        modality1 = cv2.imread(str(modality1_path))
        modality1 = cv2.cvtColor(modality1, cv2.COLOR_BGR2RGB)
        modality1 = modality1.astype(np.float32) / 255.0
        
        # 读取模态2数据 (深度图)
        modality2_path = self.modality2_paths[idx]
        modality2 = cv2.imread(str(modality2_path), cv2.IMREAD_UNCHANGED)
        modality2 = modality2.astype(np.float32)
        
        # 归一化深度图
        min_val = np.min(modality2)
        max_val = np.max(modality2)
        if max_val - min_val < 1e-6:
            modality2 = np.zeros_like(modality2)
        else:
            modality2 = (modality2 - min_val) / (max_val - min_val)
        
        # 添加通道维度
        #if len(modality2.shape) == 2:
        #    modality2 = np.expand_dims(modality2, axis=-1)
        
        # 读取mask
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # 检查mask是否有效
        if mask is None:
            raise ValueError(f"无法读取Mask: {mask_path}")
        if mask.size == 0:
            raise ValueError(f"Mask图像尺寸为0: {mask_path}")
        
        # 转换为uint8用于变换
        mask_uint8 = mask.astype(np.uint8)
        
        # 应用变换
        if self.modality1_transform:
            augmented = self.modality1_transform(image=modality1)
            modality1 = augmented['image']
        
        if self.modality2_transform:
            augmented = self.modality2_transform(image=modality2)
            modality2 = augmented['image']
        
        if self.mask_transform:
            augmented = self.mask_transform(image=mask_uint8)
            mask = augmented['image']
            # 转换为 long 类型 (int64)
            mask = mask.long()
        else:
            # 如果没有应用变换，确保 mask 是张量
            mask = torch.tensor(mask, dtype=torch.long)
        
        return modality1, modality2, mask
# ====================== 可视化函数 ======================
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """将标准化后的图像张量转换为可显示的numpy图像"""
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0).numpy()
    else:
        tensor = tensor.numpy()
        
    tensor = tensor * std + mean
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    return tensor

def display_multimodal_augmentations(dataset, num_samples=5):
    """显示多模态增强后的样本"""
    plt.figure(figsize=cfg.FIG_SIZE)
    
    for i in range(num_samples):
        # 随机选择样本
        idx = np.random.randint(0, len(dataset))
        modality1, modality2, mask = dataset[idx]
        
        # 处理模态1 (RGB)
        if isinstance(modality1, torch.Tensor):
            modality1 = denormalize(modality1)
        
        # 处理模态2 (深度图) - 保持单通道特性
        if isinstance(modality2, torch.Tensor):
            if modality2.dim() == 3 and modality2.shape[0] == 1:  # 单通道深度图
                modality2 = modality2.squeeze(0).numpy()  # [1, H, W] -> [H, W]
                # 归一化深度图用于显示
                min_val = np.min(modality2)
                max_val = np.max(modality2)
                if max_val > min_val:
                    modality2 = (modality2 - min_val) / (max_val - min_val)
                else:
                    modality2 = np.zeros_like(modality2)
                modality2 = (modality2 * 255).astype(np.uint8)
                modality2 = np.stack([modality2]*3, axis=-1)  # 转为3通道用于显示
            else:
                modality2 = denormalize(modality2)
        
        # 处理mask
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy().squeeze()

        colored_mask = apply_colormap(mask)
        
        # 创建叠加图像
        overlay1 = cv2.addWeighted(modality1, 0.7, colored_mask, 0.3, 0)
        overlay2 = cv2.addWeighted(modality2, 0.7, colored_mask, 0.3, 0)
        
        # 可视化
        plt.subplot(num_samples, 5, i*5 + 1)
        plt.imshow(modality1)
        plt.title("Modality 1 (RGB)")
        plt.axis('off')
        
        plt.subplot(num_samples, 5, i*5 + 2)
        plt.imshow(modality2)
        plt.title("Modality 2 (Depth)")
        plt.axis('off')
        
        plt.subplot(num_samples, 5, i*5 + 3)
        plt.imshow(colored_mask)
        plt.title("Segmentation Mask")
        plt.axis('off')
        
        plt.subplot(num_samples, 5, i*5 + 4)
        plt.imshow(overlay1)
        plt.title("Modality1 Overlay")
        plt.axis('off')
        
        plt.subplot(num_samples, 5, i*5 + 5)
        plt.imshow(overlay2)
        plt.title("Modality2 Overlay")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def apply_colormap(mask, colormap=plt.cm.tab10):
    """为单通道mask应用颜色映射"""
    normalized_mask = mask / (np.max(mask) + 1e-7)
    colored = colormap(normalized_mask)[:, :, :3]
    return (colored * 255).astype(np.uint8)

# ====================== 数据准备 ======================
def prepare_multimodal_datasets():
    """准备多模态训练和验证数据集"""
    # 获取所有模态1路径
    modality1_paths = sorted(list(cfg.MODALITY1_DIR.glob(f"*{cfg.MODALITY1_EXT}")))
    
    # 获取所有模态2路径 (假设文件名与模态1相同，扩展名不同)
    modality2_paths = []
    for m1_path in modality1_paths:
        m2_name = m1_path.stem + cfg.MODALITY2_EXT
        m2_path = cfg.MODALITY2_DIR / m2_name
        if m2_path.exists():
            modality2_paths.append(m2_path)
        else:
            print(f"警告: 找不到对应的模态2文件: {m2_path}")
    
    # 获取所有mask路径
    mask_paths = []
    for m1_path in modality1_paths:
        mask_name = m1_path.stem + "_label" + cfg.MASK_EXT
        mask_path = cfg.MASK_DIR / mask_name
        if mask_path.exists():
            mask_paths.append(mask_path)
        else:
            print(f"警告: 找不到对应的Mask文件: {mask_path}")
    
    # 确保所有列表长度一致
    min_len = min(len(modality1_paths), len(modality2_paths), len(mask_paths))
    modality1_paths = modality1_paths[:min_len]
    modality2_paths = modality2_paths[:min_len]
    mask_paths = mask_paths[:min_len]
    
    # 分割训练验证集
    train_m1, val_m1, train_m2, val_m2, train_mask, val_mask = train_test_split(
        modality1_paths, modality2_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # 创建变换管道
    # 模态1 (RGB) - 应用颜色增强
    modality1_train_transform = create_multimodal_augmentation(
        target_size=cfg.TARGET_SIZE, 
        apply_color_aug=True
    )
    
    # 模态2 (深度图) - 不应用颜色增强
    modality2_train_transform = create_multimodal_augmentation(
        target_size=cfg.TARGET_SIZE, 
        apply_color_aug=False
    )
    
    # Mask变换
    mask_train_transform = A.Compose([
        A.OneOf([
            A.RandomResizedCrop(
                size=(cfg.TARGET_SIZE, cfg.TARGET_SIZE),
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33),
                interpolation=cv2.INTER_NEAREST,  # 使用最近邻插值
                p=cfg.RANDOM_SCALE_PROB
            ),
            A.RandomCrop(height=cfg.TARGET_SIZE, width=cfg.TARGET_SIZE, p=cfg.RANDOM_CROP_PROB),
            A.Resize(height=cfg.TARGET_SIZE, width=cfg.TARGET_SIZE, p=cfg.RESIZE_PROB)
        ], p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.5),
        ToTensorV2()
    ])
    
    # 验证集变换 (不应用数据增强)
    modality1_val_transform = A.Compose([
        A.Resize(height=cfg.TARGET_SIZE, width=cfg.TARGET_SIZE),
        ToTensorV2()
    ])
    
    modality2_val_transform = A.Compose([
        A.Resize(height=cfg.TARGET_SIZE, width=cfg.TARGET_SIZE),
        ToTensorV2()
    ])
    
    mask_val_transform = A.Compose([
        A.Resize(height=cfg.TARGET_SIZE, width=cfg.TARGET_SIZE),
        ToTensorV2()
    ])
    
    # 创建数据集
    train_dataset = MultimodalSegmentationDataset(
        train_m1, train_m2, train_mask,
        modality1_train_transform, modality2_train_transform, mask_train_transform
    )
    
    val_dataset = MultimodalSegmentationDataset(
        val_m1, val_m2, val_mask,
        modality1_val_transform, modality2_val_transform, mask_val_transform
    )
    
    return train_dataset, val_dataset

# ====================== 主函数 ======================
if __name__ == "__main__":
    # 准备数据
    train_dataset, val_dataset = prepare_multimodal_datasets()
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    
    # 可视化训练增强效果
    print("\n=== 多模态训练数据增强示例 ===")
    display_multimodal_augmentations(train_dataset, cfg.VISUALIZE_SAMPLES)
    
    # 创建数据加载器示例
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    
    # 测试一个批次
    batch_m1, batch_m2, batch_masks = next(iter(train_loader))
    print(f"批次模态1尺寸: {batch_m1.shape}")
    print(f"批次模态2尺寸: {batch_m2.shape}")
    print(f"批次Mask尺寸: {batch_masks.shape}")
    print("多模态数据管道测试完成！")