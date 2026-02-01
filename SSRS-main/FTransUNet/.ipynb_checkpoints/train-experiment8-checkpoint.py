import os
import gc
import time
import itertools
import urllib.request
import psutil
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import confusion_matrix
from PIL import Image
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.utils.data as data
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
import torch.cuda.amp as amp
import torch.nn.init
from torch.utils.data import Dataset, DataLoader
from utils import *
from enhancement import MoE_ISP_Model
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from IPython.display import clear_output

# ==================== 1. 图像质量正则化损失函数 ====================

class EdgePreservationLoss(nn.Module):
    """边缘保持损失"""
    def __init__(self):
        super(EdgePreservationLoss, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, enhanced, original):
        # 确保 sobel 滤波器在正确的设备上
        sobel_x = self.sobel_x.to(enhanced.device)
        sobel_y = self.sobel_y.to(enhanced.device)
        
        # 计算原始图像的边缘
        original_edges_x = F.conv2d(original, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
        original_edges_y = F.conv2d(original, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
        original_edges = torch.sqrt(original_edges_x**2 + original_edges_y**2)

        # 计算增强图像的边缘
        enhanced_edges_x = F.conv2d(enhanced, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
        enhanced_edges_y = F.conv2d(enhanced, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
        enhanced_edges = torch.sqrt(enhanced_edges_x**2 + enhanced_edges_y**2)

        # 边缘保持损失
        edge_loss = F.l1_loss(enhanced_edges, original_edges)
        return edge_loss


class MultiTaskLoss(nn.Module):
    """多任务损失函数：分割损失 + 图像质量正则化"""
    def __init__(self, seg_weight=1.0, reg_weight=0.3, class_weights=None):
        super(MultiTaskLoss, self).__init__()
        self.seg_weight = seg_weight
        self.reg_weight = reg_weight
        
        # 分割损失
        if class_weights is not None:
            self.seg_criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.seg_criterion = nn.CrossEntropyLoss()
        
        # 图像质量正则化
        self.reg_criterion = ImageQualityRegularizer()
    
    def forward(self, enhanced_output, segmentation_output, original_image, segmentation_label):
        # 分割损失
        seg_loss = self.seg_criterion(segmentation_output, segmentation_label)
        
        # 图像质量正则化损失
        reg_loss, reg_details = self.reg_criterion(enhanced_output, original_image)
        
        # 添加数值稳定性检查
        if torch.isnan(seg_loss) or torch.isinf(seg_loss):
            print("警告: 分割损失为NaN或Inf，使用替代值")
            seg_loss = torch.tensor(1.0, device=seg_loss.device, requires_grad=True)
        
        if torch.isnan(reg_loss) or torch.isinf(reg_loss):
            print("警告: 正则化损失为NaN或Inf，使用替代值")
            reg_loss = torch.tensor(0.1, device=reg_loss.device, requires_grad=True)
        
        # 总损失
        total_loss = self.seg_weight * seg_loss + self.reg_weight * reg_loss
        
        return total_loss, seg_loss, reg_loss, reg_details

class SSIMLoss(nn.Module):
    """结构相似性损失"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        window = self.create_window(window_size, self.channel)
        self.register_buffer('window', window)
        
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        # 确保 window 在正确的设备上
        window = self.window.to(img1.device)
        
        if channel != self.channel:
            window = self.create_window(self.window_size, channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2
        
        # 添加数值稳定性保护
        denominator = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        denominator = torch.clamp(denominator, min=1e-8)  # 防止除零
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / denominator

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)



class ColorConstancyLoss(nn.Module):
    """色彩恒常性损失"""
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, enhanced):
        # 计算RGB通道之间的差异
        r = enhanced[:, 0, :, :]
        g = enhanced[:, 1, :, :]
        b = enhanced[:, 2, :, :]
        
        # 色彩恒常性约束：相邻像素的色彩应该平滑变化
        mean_rg = torch.mean(torch.abs(r - g))
        mean_gb = torch.mean(torch.abs(g - b))
        mean_br = torch.mean(torch.abs(b - r))
        
        return (mean_rg + mean_gb + mean_br) / 3.0


class ExposureControlLoss(nn.Module):
    """曝光控制损失"""
    def __init__(self, target_mean=0.6):
        super(ExposureControlLoss, self).__init__()
        self.target_mean = target_mean

    def forward(self, enhanced):
        # 计算图像的平均亮度
        mean_val = torch.mean(enhanced)
        # 目标是将平均亮度控制在合理范围内
        return torch.abs(mean_val - self.target_mean)


class ImageQualityRegularizer(nn.Module):
    """图像质量正则化器"""
    def __init__(self, weights=None):
        super(ImageQualityRegularizer, self).__init__()
        if weights is None:
            weights = {'ssim': 0.3, 'edge': 0.3, 'color': 0.2, 'exposure': 0.2}
        
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgePreservationLoss()
        self.color_loss = ColorConstancyLoss()
        self.exposure_loss = ExposureControlLoss()
        self.weights = weights
        
    def forward(self, enhanced, original):
        ssim_loss = self.ssim_loss(enhanced, original)
        edge_loss = self.edge_loss(enhanced, original)
        color_loss = self.color_loss(enhanced)
        exposure_loss = self.exposure_loss(enhanced)
        
        total_loss = (self.weights['ssim'] * ssim_loss +
                     self.weights['edge'] * edge_loss +
                     self.weights['color'] * color_loss +
                     self.weights['exposure'] * exposure_loss)
        
        return total_loss, {'ssim': ssim_loss, 'edge': edge_loss, 'color': color_loss, 'exposure': exposure_loss}


# ==================== 3. 端到端模型定义 ====================

class EndToEndModel(nn.Module):
    """端到端级联模型：MoE增强 + FTransUNet分割"""
    def __init__(self, moe_model, ftransunet_model):
        super(EndToEndModel, self).__init__()
        self.moe_enhancer = moe_model
        self.ftransunet_segmenter = ftransunet_model
        
    def forward(self, x, depth_map):
        # 第一阶段：图像增强
        enhanced_img = self.moe_enhancer(x, depth_map)
        depth_map = depth_map.squeeze()
        # 第二阶段：语义分割
        segmentation_output = self.ftransunet_segmenter(enhanced_img, depth_map)
        
        return enhanced_img, segmentation_output

# ==================== 4. 数据集类 ====================

class EndToEndDataset(Dataset):
    """端到端训练数据集 - 修复版本"""
    def __init__(self, ids, image_folder, depth_folder, label_folder, transform=None, scales=[0.75, 1.0, 1.25]):
        self.ids = ids
        self.image_folder = image_folder
        self.depth_folder = depth_folder
        self.label_folder = label_folder
        self.transform = transform
        self.scales = scales
        self.window_size = (256, 256)
        self.context_margin = 64
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        id = self.ids[idx]
        
        # 随机选择尺度
        scale = np.random.choice(self.scales)
        
        # 加载原始图像
        img = Image.open(self.image_folder.format(id)).convert('RGB')
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
        
        # 加载深度图
        dsm = Image.open(self.depth_folder.format(id)).convert('L')
        dsm = dsm.resize((int(dsm.width * scale), int(dsm.height * scale)), Image.BILINEAR)
        
        # 加载分割标签
        label = Image.open(self.label_folder.format(id)).convert('L')
        label = label.resize((int(label.width * scale), int(label.height * scale)), Image.NEAREST)
        
        # 转换为数组
        img_arr = np.array(img, dtype=np.float32) / 255.0
        dsm_arr = np.array(dsm, dtype=np.float32)
        label_arr = np.array(label, dtype=np.uint8)

        if np.isnan(img_arr).any() or np.isinf(img_arr).any():
            print(f"警告: 图像 {id} 包含NaN或Inf值")
            img_arr = np.nan_to_num(img_arr)
        
        if np.isnan(dsm_arr).any() or np.isinf(dsm_arr).any():
            print(f"警告: 深度图 {id} 包含NaN或Inf值")
            dsm_arr = np.nan_to_num(dsm_arr)
        
        # 归一化DSM
        dsm_arr = self.normalize_dsm(dsm_arr)
        
        # 随机裁剪
        h, w = img_arr.shape[:2]
        crop_size = self.window_size[0] + 2 * self.context_margin
        
        if w < crop_size or h < crop_size:
            img_patch = img_arr
            dsm_patch = dsm_arr
            label_patch = label_arr
        else:
            x = np.random.randint(0, max(0, w - crop_size))
            y = np.random.randint(0, max(0, h - crop_size))
            
            img_patch = img_arr[y:y+crop_size, x:x+crop_size]
            dsm_patch = dsm_arr[y:y+crop_size, x:x+crop_size]
            label_patch = label_arr[y:y+crop_size, x:x+crop_size]
        
        # 转换为PIL图像进行大小调整
        img_patch_pil = Image.fromarray((img_patch * 255).astype(np.uint8))
        dsm_patch_pil = Image.fromarray((dsm_patch * 255).astype(np.uint8))
        label_patch_pil = Image.fromarray(label_patch.astype(np.uint8))
        
        # 调整到模型输入大小
        img_patch_resized = img_patch_pil.resize(self.window_size, Image.BILINEAR)
        dsm_patch_resized = dsm_patch_pil.resize(self.window_size, Image.BILINEAR)
        label_patch_resized = label_patch_pil.resize(self.window_size, Image.NEAREST)
        
        # 转换为张量 - 在CPU上创建
        if self.transform:
            img_tensor = self.transform(img_patch_resized)
        else:
            img_tensor = transforms.ToTensor()(img_patch_resized)
        
        # 关键修复：在CPU上创建张量，不指定device参数
        dsm_tensor = torch.tensor(np.array(dsm_patch_resized), dtype=torch.float32).unsqueeze(0) / 255.0
        label_tensor = torch.tensor(np.array(label_patch_resized), dtype=torch.long)
        
        return img_tensor, dsm_tensor, label_tensor
    
    def normalize_dsm(self, dsm_arr):
        """DSM归一化函数"""
        epsilon = 1e-8
        min_val = np.min(dsm_arr)
        max_val = np.max(dsm_arr)
        
        if max_val - min_val < epsilon:
            return np.zeros_like(dsm_arr)
        
        clipped_dsm = np.clip(dsm_arr, min_val, min_val + 10*(max_val-min_val))
        return (clipped_dsm - min_val) / (max_val - min_val + epsilon)

# ==================== 5. 渐进式训练策略 ====================

def freeze_segmentation_model(model):
    """冻结分割模型参数"""
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_segmentation_model(model):
    """解冻分割模型参数"""
    for param in model.parameters():
        param.requires_grad = True

def create_progressive_optimizer(model, stage, moe_lr=1e-5, seg_lr=1e-6):
    """创建渐进式优化器 - 修复版本"""
    if stage == 1:  # 阶段1：只训练MoE
        params = [{'params': model.moe_enhancer.parameters(), 'lr': moe_lr}]
    elif stage == 2:  # 阶段2：联合训练，侧重MoE
        params = [
            {'params': model.moe_enhancer.parameters(), 'lr': moe_lr},
            {'params': model.ftransunet_segmenter.parameters(), 'lr': seg_lr}
        ]
    else:  # 阶段3：平衡训练
        params = model.parameters()
    
    optimizer = optim.AdamW(params, lr=moe_lr, weight_decay=1e-4, betas=(0.9, 0.999))
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )
    
    return optimizer, scheduler  # 返回两个值

    
# ==================== 6. 训练函数 ====================

def train_progressive(model, train_loader, val_loader, device, save_dir, total_epochs=200):
    """渐进式训练函数 - 修复版本"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练阶段配置
    stage_configs = [
        {'name': '阶段1-MoE预训练', 'epochs': 40, 'stage': 1, 'reg_weight': 0.5},
        {'name': '阶段2-联合训练侧重MoE', 'epochs': 60, 'stage': 2, 'reg_weight': 0.3},
        {'name': '阶段3-平衡训练', 'epochs': 100, 'stage': 3, 'reg_weight': 0.1}
    ]
    
    best_mIoU = 0.0
    start_epoch = 0
    train_history = []
    
    for stage_config in stage_configs:
        stage_name = stage_config['name']
        stage_epochs = stage_config['epochs']
        stage = stage_config['stage']
        reg_weight = stage_config['reg_weight']
        
        print(f"\n{'='*50}")
        print(f"开始{stage_name}")
        print(f"{'='*50}")
        
        # 配置模型参数训练状态
        if stage == 1:
            freeze_segmentation_model(model.ftransunet_segmenter)
        else:
            unfreeze_segmentation_model(model.ftransunet_segmenter)
        
        # 创建优化器和调度器
        optimizer, scheduler = create_progressive_optimizer(model, stage)  # 现在返回两个值
        
        # 创建损失函数
        criterion = MultiTaskLoss(seg_weight=1.0, reg_weight=reg_weight).to(device)
        
        # 创建梯度scaler
        scaler = amp.GradScaler()
        
        # 阶段训练
        for epoch in range(stage_epochs):
            current_epoch = start_epoch + epoch + 1
            
            # 训练一个epoch
            train_loss, train_details = train_epoch(
                model, train_loader, optimizer, criterion, device, scaler, stage  # 传递optimizer而不是元组
            )
            
            # 验证
            if (current_epoch) % 5 == 0:
                val_metrics = validate_model(model, val_loader, criterion, device)
                
                # 使用验证损失调整学习率
                scheduler.step(val_metrics['val_loss'])
                
                # 记录训练历史
                history_entry = {
                    'epoch': current_epoch,
                    'stage': stage,
                    'train_loss': train_loss,
                    'train_details': train_details,
                    'val_metrics': val_metrics
                }
                train_history.append(history_entry)
                
                print(f"Epoch {current_epoch}/{total_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val mIoU: {val_metrics['mIoU']:.4f}")
                
                # 保存最佳模型
                if val_metrics['mIoU'] > best_mIoU:
                    best_mIoU = val_metrics['mIoU']
                    torch.save({
                        'epoch': current_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_mIoU': best_mIoU,
                        'train_history': train_history
                    }, os.path.join(save_dir, 'best_model.pth'))
                    print(f"✅ 保存最佳模型，mIoU: {best_mIoU:.4f}")
            
            # 保存检查点
            if (current_epoch) % 10 == 0:
                torch.save({
                    'epoch': current_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_history': train_history
                }, os.path.join(save_dir, f'checkpoint_epoch_{current_epoch}.pth'))
        
        start_epoch += stage_epochs
    
    return train_history, best_mIoU

def train_epoch(model, train_loader, optimizer, criterion, device, scaler, stage):
    """训练一个epoch - 修复版本"""
    model.train()
    total_loss = 0.0
    seg_loss_total = 0.0
    reg_loss_total = 0.0
    reg_details = {'ssim': 0.0, 'edge': 0.0, 'color': 0.0, 'exposure': 0.0}
    
    progress_bar = tqdm(train_loader, desc=f'训练中 (阶段{stage})')
    
    for batch_idx, (images, depths, labels) in enumerate(progress_bar):
        # 数据移动到GPU
        images = images.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()  # 现在optimizer是单个对象，不是元组
        
        # 混合精度训练
        with amp.autocast():
            enhanced_output, seg_output = model(images, depths)
            loss, seg_loss, reg_loss, details = criterion(
                enhanced_output, seg_output, images, labels
            )
            
            # 检查损失值
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 批次 {batch_idx} 总损失为NaN或Inf，跳过该批次")
                continue

        # 反向传播
        scaler.scale(loss).backward()
        
        # 先 unscale（为梯度裁剪准备）
        try:
            scaler.unscale_(optimizer)
        except Exception as e:
            # 记录并继续，后面会尝试 fallback 步骤
            print(f"⚠️ scaler.unscale_ 失败: {e}")
        
        # 梯度裁剪（在 unscale 后）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 检查并修复 NaN/Inf 梯度（保守）
        grad_has_nan = False
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                print(f"警告: 参数 {name} 的梯度包含 NaN/Inf，已修复")
                p.grad.data = torch.nan_to_num(p.grad.data)
                grad_has_nan = True
        
        # 安全调用 scaler.step，若断言失败则 fallback 到普通 optimizer.step()
        try:
            scaler.step(optimizer)   # 正常路径（会做 finite 检查）
        except AssertionError as ae:
            # 典型错误: "No inf checks were recorded for this optimizer."
            print(f"⚠️ scaler.step 断言失败: {ae}，使用 fallback optimizer.step()")
            # 直接用普通 step（注意这里的风险：若存在 Inf/NaN 梯度可能破坏权重）
            optimizer.step()
        except Exception as e:
            print(f"⚠️ scaler.step 其他异常: {e}, 使用 fallback optimizer.step()")
            optimizer.step()
        
        # 最后更新 scaler（无论是否使用 fallback，都更新 scaler）
        try:
            scaler.update()
        except Exception as e:
            # 如果 scaler.update 也出问题，记录下来但不要中断训练
            print(f"⚠️ scaler.update 失败: {e}")

        
        # 记录损失
        total_loss += loss.item()
        seg_loss_total += seg_loss.item()
        reg_loss_total += reg_loss.item()
        
        for key in reg_details.keys():
            reg_details[key] += details[key].item()
        
        # 更新进度条
        if batch_idx % 10 == 0:
            progress_bar.set_postfix({
                '总损失': f'{loss.item():.4f}',
                '分割损失': f'{seg_loss.item():.4f}',
                '正则损失': f'{reg_loss.item():.4f}'
            })
        
        # 清理内存
        del images, depths, labels, enhanced_output, seg_output, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 计算平均损失
    num_batches = len(train_loader)
    if num_batches == 0:
        print("警告: 该epoch没有有效批次")
        return 1.0, {'seg_loss': 1.0, 'reg_loss': 0.1, 'reg_details': reg_details}
    
    avg_total_loss = total_loss / num_batches
    avg_seg_loss = seg_loss_total / num_batches
    avg_reg_loss = reg_loss_total / num_batches
    
    for key in reg_details.keys():
        reg_details[key] /= num_batches
    
    train_details = {
        'seg_loss': avg_seg_loss,
        'reg_loss': avg_reg_loss,
        'reg_details': reg_details
    }
    
    return avg_total_loss, train_details

def validate_model(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for images, depths, labels in tqdm(val_loader, desc='验证中'):
            images = images.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            enhanced_output, seg_output = model(images, depths)
            loss, _, _, _ = criterion(enhanced_output, seg_output, images, labels)
            total_loss += loss.item()
            
            # 获取预测结果
            _, preds = torch.max(seg_output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())
    
    # 计算指标
    pa, cpa, mpa, mIoU = calculate_metrics(
        np.concatenate([p.ravel() for p in all_preds]),
        np.concatenate([p.ravel() for p in all_gts])
    )
    
    return {
        'val_loss': total_loss / len(val_loader),
        'pa': pa,
        'mpa': mpa,
        'mIoU': mIoU
    }

# ==================== 7. 测试函数 ====================

def test_model(model, test_loader, device, output_dir):
    """测试模型"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for i, (images, depths, labels) in enumerate(tqdm(test_loader, desc='测试中')):
            images = images.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            labels_cpu = labels.cpu().numpy()
            
            enhanced_output, seg_output = model(images, depths)
            _, preds = torch.max(seg_output, 1)
            preds = preds.cpu().numpy()
            
            all_preds.extend(preds)
            all_gts.extend(labels_cpu)
            
            # 可视化部分结果
            if i < 5:
                visualize_results(images, enhanced_output, preds, labels_cpu, output_dir, i)
    
    # 计算指标
    pa, cpa, mpa, mIoU = calculate_metrics(
        np.concatenate([p.ravel() for p in all_preds]),
        np.concatenate([p.ravel() for p in all_gts])
    )
    
    # 保存结果
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"PA: {pa:.4f}\n")
        f.write(f"MPA: {mpa:.4f}\n")
        f.write(f"mIoU: {mIoU:.4f}\n")
    
    print(f"测试完成 - PA: {pa:.4f}, MPA: {mpa:.4f}, mIoU: {mIoU:.4f}")
    
    return all_preds, all_gts, {'pa': pa, 'mpa': mpa, 'mIoU': mIoU}

def visualize_results(images, enhanced, predictions, labels, output_dir, batch_idx):
    """可视化结果"""
    for i in range(min(3, images.shape[0])):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原始图像
        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        axes[0, 0].imshow(np.clip(img_np, 0, 1))
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
        
        # 增强后图像
        enhanced_np = enhanced[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, 1].imshow(np.clip(enhanced_np, 0, 1))
        axes[0, 1].set_title('增强后图像')
        axes[0, 1].axis('off')
        
        # 预测结果
        axes[1, 0].imshow(predictions[i], cmap='tab20')
        axes[1, 0].set_title('预测分割')
        axes[1, 0].axis('off')
        
        # 真实标签
        axes[1, 1].imshow(labels[i], cmap='tab20')
        axes[1, 1].set_title('真实标签')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'result_batch{batch_idx}_sample{i}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

# ==================== 8. 工具函数 ====================

def calculate_metrics(y_pred, y_true):
    """计算分割指标"""
    cm = confusion_matrix(y_true, y_pred)
    pa = np.diag(cm).sum() / cm.sum()
    cpa = np.diag(cm) / cm.sum(axis=1)
    mpa = np.nanmean(cpa)
    iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    mIoU = np.nanmean(iou)
    return pa, cpa, mpa, mIoU

def read_txt_to_list(file_path):
    """读取文本文件到列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [line.strip() for line in lines if line.strip()]
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return []

def plot_training_history(train_history, save_dir):
    """绘制训练历史图表"""
    if not train_history:
        return
    
    epochs = [entry['epoch'] for entry in train_history]
    train_losses = [entry['train_loss'] for entry in train_history]
    val_mIoUs = [entry['val_metrics']['mIoU'] for entry in train_history]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失曲线')
    ax1.legend()
    ax1.grid(True)
    
    # mIoU曲线
    ax2.plot(epochs, val_mIoUs, 'r-', label='验证mIoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU')
    ax2.set_title('验证mIoU曲线')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 训练历史图表已保存")

def initialize_weights(model):
    """初始化模型权重"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.normal_(param, mean=0, std=0.01)
        elif 'bias' in name:
            nn.init.constant_(param, 0)
    
    print("✅ 模型权重初始化完成")

# ==================== 9. 主程序 ====================

if __name__ == "__main__":
    # 1. 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 2. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 3. 读取数据ID
    train_ids = read_txt_to_list("train1.txt")
    val_ids = read_txt_to_list("val1.txt")
    test_ids = read_txt_to_list("test1.txt")
    print(f"训练集大小: {len(train_ids)}")
    print(f"验证集大小: {len(val_ids)}")
    print(f"测试集大小: {len(test_ids)}")
    
    # 4. 创建数据集
    DATA_FOLDER = "/root/autodl-tmp/dataset/cropped_Image/{}.jpg"
    DSM_FOLDER = "/root/autodl-tmp/dataset/cropped_Depth/{}.png"
    LABEL_FOLDER = "/root/autodl-tmp/dataset/cropped_Label/{}_label.png"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 不传递device参数给Dataset
    train_dataset = EndToEndDataset(train_ids, DATA_FOLDER, DSM_FOLDER, LABEL_FOLDER, transform)
    val_dataset = EndToEndDataset(val_ids, DATA_FOLDER, DSM_FOLDER, LABEL_FOLDER, transform)
    test_dataset = EndToEndDataset(test_ids, DATA_FOLDER, DSM_FOLDER, LABEL_FOLDER, transform)
    
    # 5. 创建数据加载器
    BATCH_SIZE = 8
    NUM_WORKERS = 4  # 可以根据需要调整
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    # 6. 初始化模型
    print("\n初始化模型...")
    moe_model = MoE_ISP_Model(num_experts=3).to(device)

    initialize_weights(moe_model)
    
    # 初始化FTransUNet模型
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 16
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(256 / 16), int(256 / 16))
    
    ftransunet_model = ViT_seg(config_vit, img_size=256, num_classes=16).to(device)
    
    # 加载预训练权重（如果有）
    try:
        ftransunet_model.load_from(weights=np.load(config_vit.pretrained_path, allow_pickle=True))
        print("✅ FTransUNet预训练权重加载成功")
    except Exception as e:
        print(f"⚠️ FTransUNet预训练权重加载失败: {e}")
        print("将使用随机初始化的权重")
    
    # 创建端到端模型
    model = EndToEndModel(moe_model, ftransunet_model).to(device)
    
    # 打印模型参数
    moe_params = sum(p.numel() for p in moe_model.parameters())
    seg_params = sum(p.numel() for p in ftransunet_model.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"MoE模型参数: {moe_params:,}")
    print(f"分割模型参数: {seg_params:,}")
    print(f"总参数: {total_params:,}")
    
    # 7. 开始训练
    save_dir = "./checkpoints"
    total_epochs = 200
    
    print(f"\n开始训练，总轮数: {total_epochs}")
    print(f"结果将保存到: {save_dir}")
    
    train_history, best_mIoU = train_progressive(
        model, train_loader, val_loader, device, save_dir, total_epochs
    )
    
    print(f"\n训练完成！最佳mIoU: {best_mIoU:.4f}")
    
    # 8. 测试模型
    print("\n开始测试...")
    
    # 加载最佳模型
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 加载最佳模型 (epoch {checkpoint['epoch']}, mIoU: {checkpoint['best_mIoU']:.4f})")
    else:
        print("⚠️ 未找到最佳模型，使用最终模型进行测试")
    
    # 测试
    test_output_dir = os.path.join(save_dir, 'test_results')
    all_preds, all_gts, test_metrics = test_model(
        model, test_loader, device, test_output_dir
    )
    
    # 9. 结果分析
    print("\n" + "="*50)
    print("最终结果汇总")
    print("="*50)
    print(f"测试集大小: {len(test_ids)}")
    print(f"PA: {test_metrics['pa']:.4f}")
    print(f"MPA: {test_metrics['mpa']:.4f}")
    print(f"mIoU: {test_metrics['mIoU']:.4f}")
    
    # 保存训练历史
    plot_training_history(train_history, save_dir)
    
    print(f"\n所有任务完成！结果保存在: {save_dir}")