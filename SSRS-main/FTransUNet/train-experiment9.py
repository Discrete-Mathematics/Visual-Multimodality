# -*- coding: utf-8 -*-
"""
端到端 MoE-Enhancer + FTransUNet 训练 & 测试
（断点续训 + 内存安全 + PyTorch 2.6 兼容版）
"""
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
import torch.amp as amp
import torch.nn.init
from torch.utils.data import Dataset, DataLoader
from utils import *
from enhancement import MoE_ISP_Model
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from IPython.display import clear_output
import json
from datetime import datetime
import cv2

# ==================== 实验记录工具类 ====================
class ExperimentLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.experiment_start_time = datetime.now()
        self.experiment_id = self.experiment_start_time.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(save_dir, f"experiment_{self.experiment_id}.log")
        self.config_file = os.path.join(save_dir, f"experiment_{self.experiment_id}_config.json")
        self.results_file = os.path.join(save_dir, f"experiment_{self.experiment_id}_results.json")
        os.makedirs(save_dir, exist_ok=True)

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        print(log_entry.strip())
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def save_config(self, config):
        config["experiment_id"] = self.experiment_id
        config["start_time"] = self.experiment_start_time.strftime("%Y-%m-%d %H:%M:%S")
        config["device"] = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        self.log(f"实验配置已保存至 {self.config_file}")

    def save_results(self, results):
        results["experiment_id"] = self.experiment_id
        results["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results["duration"] = str(datetime.now() - self.experiment_start_time)
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        self.log(f"实验结果已保存至 {self.results_file}")

# ==================== 2. 端到端模型 ====================
class EndToEndModel(nn.Module):
    def __init__(self, moe_model, ftransunet_model):
        super().__init__()
        self.moe_enhancer = moe_model
        self.ftransunet_segmenter = ftransunet_model

    def forward(self, x, depth_map):
        if depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(1)
        elif depth_map.dim() == 4 and depth_map.size(1) != 1:
            depth_map = depth_map.mean(dim=1, keepdim=True)
        enhanced_img = self.moe_enhancer(x, depth_map)
        enhanced_img = torch.clamp(enhanced_img, 0.0, 1.0)
        segmentation_output = self.ftransunet_segmenter(enhanced_img, depth_map.squeeze(1))
        return enhanced_img, segmentation_output
def validate_model(model, val_loader, criterion, device, logger=None):
    """验证模型，增加logger参数"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_gts = []
    
    # 使用logger或默认打印
    log_func = logger.log if logger else print
    
    # 【内存/显存优化】在验证开始前清理一次 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for images, depths, labels in tqdm(val_loader, desc='验证中'):
            # 1. 数据加载
            images = images.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 2. 前向传播
            enhanced_output, seg_output = model(images, depths)
            
            # 3. 计算损失 (使用 seg_criterion 是正确的，因为验证通常只关注最终性能)
            # 确保 criterion.seg_criterion 是一个有效的 PyTorch 损失函数
            seg_loss = criterion.seg_criterion(seg_output, labels)
            
            # 【内存/显存优化】使用 .item() 记录数值，避免保留损失张量的引用
            total_loss += seg_loss.item()
            
            # 4. 获取预测结果
            _, preds = torch.max(seg_output, 1)
            
            # 【内存/显存优化】使用 .detach().cpu().numpy()
            # .detach() 确保脱离计算图 (即使在 no_grad() 中也是好习惯)
            # .cpu() 将张量从 GPU 移到 CPU
            all_preds.extend(preds.detach().cpu().numpy())
            all_gts.extend(labels.detach().cpu().numpy()) # labels也应该被转移和处理
            
            # 5. 【内存/显存优化】显式删除批次内的 GPU 张量，防止在验证过程中积累
            del images, depths, labels, enhanced_output, seg_output, seg_loss, preds
            
            # 【内存/显存优化】可选：每隔 N 批次清理一次缓存
            # if batch_idx % 10 == 0 and torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            
    # 【内存/显存优化】验证循环结束后，清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 计算指标
    # 注意：这里假设 calculate_metrics 函数能够处理扁平化的 numpy 数组
    try:
        pa, cpa, mpa, mIoU = calculate_metrics(
            np.concatenate([p.ravel() for p in all_preds]),
            np.concatenate([g.ravel() for g in all_gts])
        )
    except Exception as e:
        log_func(f"❌ 指标计算失败: {e}. 可能原因是 all_preds/all_gts 为空或维度不匹配。")
        # 返回默认值以防止程序崩溃
        return { 'val_loss': total_loss / len(val_loader) if len(val_loader) > 0 else 0,
                 'pa': 0, 'mpa': 0, 'mIoU': 0, 'cpa': [] }
        
    
    log_func(f"验证指标 - PA: {pa:.4f}, MPA: {mpa:.4f}, mIoU: {mIoU:.4f}")
    
    return {
        'val_loss': total_loss / len(val_loader),
        'pa': pa,
        'mpa': mpa,
        'mIoU': mIoU,
        'cpa': cpa.tolist()  # 转为列表以便序列化
    }

# ==================== 4. 渐进式训练策略 ====================
def freeze_segmentation_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_segmentation_model(model):
    for param in model.parameters():
        param.requires_grad = True

def create_progressive_optimizer(model, stage, moe_lr=1e-7, seg_lr=1e-7):
    if stage == 1:
        params = [{'params': model.moe_enhancer.parameters(), 'lr': moe_lr}]
    elif stage == 2:
        params = [
            {'params': model.moe_enhancer.parameters(), 'lr': moe_lr},
            {'params': model.ftransunet_segmenter.parameters(), 'lr': seg_lr}
        ]
    else:
        params = [
            {'params': model.moe_enhancer.parameters(), 'lr': moe_lr},
            {'params': model.ftransunet_segmenter.parameters(), 'lr': seg_lr}
        ]
    optimizer = optim.AdamW(params, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / 10.0))
    return optimizer, scheduler


# ==================== 5. 训练函数 ====================
def train_progressive(model, train_loader, val_loader,
                      optimizer, scheduler, device, save_dir, total_epochs=200,
                      logger=None, start_epoch=0, best_mIoU=0.0, history=None):
    os.makedirs(save_dir, exist_ok=True)
    if logger is None:
        logger = ExperimentLogger(save_dir)
    history = history or []

    stage_configs = [
        {'name': '阶段1-MoE预训练', 'epochs': 0, 'stage': 1, 'reg_weight': 0.5},
        {'name': '阶段2-联合训练侧重MoE', 'epochs': 0, 'stage': 2, 'reg_weight': 0.3},
        {'name': '阶段3-平衡训练', 'epochs': 350, 'stage': 3, 'reg_weight': 0.01}
    ]

    epochs_done = 0
    for cfg in stage_configs:
        name, epochs, stage, reg_w = cfg['name'], cfg['epochs'], cfg['stage'], cfg['reg_weight']
        if start_epoch >= epochs_done + epochs:          # 已完成的阶段直接跳过
            epochs_done += epochs
            logger.log(f'跳过 {name}，已完成 {epochs} 轮')
            continue
        logger.log(f'\n{"="*50}\n开始 {name}\n{"="*50}')

        # 阶段优化器 / 调度器 / 损失
        if optimizer is None or scheduler is None:      # 第一次或续训时重新创建
            optimizer, scheduler = create_progressive_optimizer(model, stage)
        criterion = MultiTaskLoss(seg_weight=1.0, reg_weight=reg_w).to(device)

        # 参数冻结
        freeze_segmentation_model(model.ftransunet_segmenter) if stage == 1 else unfreeze_segmentation_model(model.ftransunet_segmenter)

        for epoch in range(epochs):
            curr_epoch = epochs_done + epoch + 1
            if curr_epoch < start_epoch:
                continue
            # ---- 训练 ----
            train_loss, train_details = train_epoch(model, train_loader, optimizer, criterion, device, stage, logger)
            # ---- 验证 ----
            if curr_epoch % 5 == 0:
                val_metrics = validate_model(model, val_loader, criterion, device, logger)
                scheduler.step(val_metrics['val_loss'])
                history.append({'epoch': curr_epoch, 'stage': stage,
                                'train_loss': train_loss, 'train_details': train_details,
                                'val_metrics': val_metrics, 'lr': optimizer.param_groups[0]['lr']})
                is_best = val_metrics['mIoU'] > best_mIoU
                best_mIoU = max(best_mIoU, val_metrics['mIoU'])
                save_checkpoint({'epoch': curr_epoch, 'model_state_dict': model.state_dict(),
                                 'optimizer_state_dict': optimizer.state_dict(),
                                 'scheduler_state_dict': scheduler.state_dict(),
                                 'best_mIoU': best_mIoU, 'history': history},
                                is_best, save_dir)
                logger.log(f'Epoch {curr_epoch}/{total_epochs} | '
                           f'Train Loss: {train_loss:.4f} | Val mIoU: {val_metrics["mIoU"]:.4f}')
            if curr_epoch % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        epochs_done += epochs
        # 阶段结束清理
        del criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return history, best_mIoU
class MultiTaskLoss(nn.Module):
    def __init__(self, seg_weight=1.0, reg_weight=0.3, class_weights=None):
        super(MultiTaskLoss, self).__init__()
        self.seg_weight = seg_weight
        self.reg_weight = reg_weight
        
        # 实例化损失函数作为类成员，这是标准做法，不会泄漏
        if class_weights is not None:
            # 确保 class_weights 在正确的设备上（如果它是一个张量）
            self.register_buffer('class_weights_tensor', class_weights)
            self.seg_criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor, ignore_index=255) 
        else:
            self.seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
            
        # 假设 ImageQualityRegularizer 内部也做了内存安全处理
        self.reg_criterion = ImageQualityRegularizer()
        
        # 【内存安全修复】不存储任何 GPU 张量作为 self 的成员
        # 例如：self.last_input = None (如果存储了，会泄漏)

    def forward(self, enhanced_output, segmentation_output, original_image, segmentation_label):
        
        # 1. 检查输入是否存在 NaNs (防御性编程)
        if torch.isnan(enhanced_output).any() or torch.isinf(enhanced_output).any():
            print("Warning: Enhanced output contains NaNs or Infs!")
            # 如果增强输出 NaN/Inf，损失计算可能会崩溃，返回 0 损失并让外部 continue 
            # (这是一种处理方式，但最好在外部 train_epoch 中捕获)
            return torch.tensor(0.0, device=enhanced_output.device), \
                   torch.tensor(0.0, device=enhanced_output.device), \
                   torch.tensor(0.0, device=enhanced_output.device), \
                   {'ssim': torch.tensor(0.0), 'edge': torch.tensor(0.0), 'color': torch.tensor(0.0), 'exposure': torch.tensor(0.0)}
            
        # 2. 分割损失
        seg_loss = self.seg_criterion(segmentation_output, segmentation_label)
        
        # 3. 正则损失
        # 内存泄漏最可能发生在 ImageQualityRegularizer 的复杂计算中
        reg_loss, reg_details = self.reg_criterion(enhanced_output, original_image)
        
        # 4. 总损失
        total_loss = self.seg_weight * seg_loss + self.reg_weight * reg_loss
        
        # 【内存/显存优化】: 在 forward 函数末尾，**不进行任何清理**！
        # 清理应该留给外部的 train_epoch 函数，因为它需要 total_loss 进行 .backward()，
        # 并且需要 seg_loss, reg_loss, reg_details 来记录统计信息。
        # 在这里执行 del/empty_cache() 会破坏计算图。
        
        return total_loss, seg_loss, reg_loss, reg_details
def train_epoch(model, train_loader, optimizer, criterion, device, stage, logger=None):
    model.train()
    total_loss = 0.0
    seg_loss_total = 0.0
    reg_loss_total = 0.0
    
    # 初始化统计字典
    # 注意：统计字典的值是 Python float，因此不会造成 PyTorch 内存泄漏
    reg_details_accum = {'ssim': 0.0, 'edge': 0.0, 'color': 0.0, 'exposure': 0.0}
    
    log_func = logger.log if logger else print
    progress_bar = tqdm(train_loader, desc=f'训练中 (阶段{stage})')
    
    # 开启异常检测（速度会慢），找到错误后请注释掉
    torch.autograd.set_detect_anomaly(True) 
    
    # 【内存/显存优化】在每个 Epoch 开始前清理一次，确保环境干净
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for batch_idx, (images, depths, labels) in enumerate(progress_bar):
        # 初始化 batch_loss，用于本次循环的统计，防止引用外部变量
        batch_loss = None 
        
        try:
            # 数据转移到设备
            images = images.to(device, non_blocking=True)
            depths = depths = depths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad() 
            
            # 前向传播
            enhanced_output, seg_output = model(images, depths)
            
            # 检查模型输出是否已经是 NaN 了
            if torch.isnan(enhanced_output).any():
                log_func(f"❌ 批次 {batch_idx}: 模型增强输出包含 NaN，跳过此批次并清理显存")
                # 【内存/显存优化】在 continue 之前清理本次批次的 GPU 张量
                del images, depths, labels, enhanced_output, seg_output 
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
    
            # 计算损失
            batch_loss, seg_loss, reg_loss, details = criterion(
                enhanced_output, seg_output, images, labels
            )
            
            # 检查损失是否为 NaN
            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                log_func(f"❌ 批次 {batch_idx}: 损失计算结果为 NaN/Inf (Seg: {seg_loss.item():.4f}, Reg: {reg_loss.item():.4f})")
                # 【内存/显存优化】在 continue 之前清理本次批次的 GPU 张量
                del images, depths, labels, enhanced_output, seg_output, batch_loss, seg_loss, reg_loss, details
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            
            # 反向传播
            batch_loss.backward()
            
            # 梯度裁剪 (防止梯度爆炸导致的 NaN)
            gradient_is_nan = False
            for param in model.parameters():
                if param.grad is not None:
                    # 【内存/显存优化】检查梯度是否为 NaN/Inf
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print("⚠️ 发现梯度 NaN 或 Inf，跳过此步更新！")
                        gradient_is_nan = True
                        break
            
            if gradient_is_nan:
                optimizer.zero_grad() # 清空梯度
                # 【内存/显存优化】跳过 step() 前的显存清理
                del images, depths, labels, enhanced_output, seg_output, batch_loss, seg_loss, reg_loss, details
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue # 跳过 step()
                
            # 只有梯度正常才裁剪和更新
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1) 
            
            # 【注意】原代码里调用了两次 optimizer.step()，这是不正确的。
            # 应该只调用一次。我删除了一次调用。
            optimizer.step()
            
            # --- 只有成功更新才记录数据 ---
            # 【内存/显存优化】使用 .item() 转换为 Python float，防止泄漏
            total_loss += batch_loss.item()
            seg_loss_total += seg_loss.item()
            reg_loss_total += reg_loss.item()
            
            for key in details.keys():
                reg_details_accum[key] += details[key].item() # 使用 .item()
            
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'Total': f'{batch_loss.item():.4f}',
                    'Seg': f'{seg_loss.item():.4f}',
                    'Reg': f'{reg_loss.item():.4f}'
                })
        
        except RuntimeError as e:
            # 【内存/显存优化】捕获运行时错误（如 CUDA OOM），并清理显存
            log_func(f"❌ 运行时错误: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
            
        finally:
            # 【内存/显存优化】无论是否发生错误，确保在批次结束时删除张量
            # 避免局部变量在循环中不断积累引用。
            # 如果 batch_loss 为 None（发生在 continue 之前），del 会报错，所以只删除存在的变量。
            if 'images' in locals(): del images
            if 'depths' in locals(): del depths
            if 'labels' in locals(): del labels
            if 'enhanced_output' in locals(): del enhanced_output
            if 'seg_output' in locals(): del seg_output
            if batch_loss is not None: del batch_loss
            if 'seg_loss' in locals(): del seg_loss
            if 'reg_loss' in locals(): del reg_loss
            if 'details' in locals(): del details

            # 【内存/显存优化】每隔固定批次清理一次，例如每100批次
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 关闭异常检测
    torch.autograd.set_detect_anomaly(False)

    num_batches = len(train_loader)
    # 防止除零错误
    if num_batches == 0: 
        return 0, {}
        
    avg_total_loss = total_loss / num_batches
    avg_seg_loss = seg_loss_total / num_batches
    avg_reg_loss = reg_loss_total / num_batches
    
    # 计算平均 details
    avg_reg_details = {k: v / num_batches for k, v in reg_details_accum.items()}
    
    train_details = {
        'seg_loss': avg_seg_loss,
        'reg_loss': avg_reg_loss,
        'reg_details': avg_reg_details
    }
    
    # 【内存/显存优化】Epoch 结束时进行最终清理
    # 注意：del seg_output, enhanced_output, loss 已经在循环内部处理了。
    if torch.cuda.is_available():
        # 注意：这里我将 torchvision.cuda.empty_cache() 修正为 torch.cuda.empty_cache() 
        # 因为 torchvision 模块中没有 cuda.empty_cache() 函数
        torch.cuda.empty_cache()
        
    return avg_total_loss, train_details

class ExposureControlLoss(nn.Module):
    """曝光控制损失"""
    def __init__(self, target_mean=0.5):
        super(ExposureControlLoss, self).__init__()
        self.target_mean = target_mean

    def forward(self, enhanced):
        # 计算图像的平均亮度
        mean_val = torch.mean(enhanced)
        # 目标是将平均亮度控制在合理范围内
        return torch.abs(mean_val - self.target_mean)
class EdgePreservationLoss(nn.Module):
    """边缘保持损失 - 修复版"""
    def __init__(self):
        super(EdgePreservationLoss, self).__init__()
        # ... (保持初始化代码不变)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.epsilon = 1e-6  # 添加极小值防止 sqrt(0)

    def forward(self, enhanced, original):
        sobel_x = self.sobel_x.to(enhanced.device)
        sobel_y = self.sobel_y.to(enhanced.device)
        
        # 原始图像边缘
        original_edges_x = F.conv2d(original, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
        original_edges_y = F.conv2d(original, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
        # 关键修改：在 sqrt 内部加入 epsilon
        original_edges = torch.sqrt(original_edges_x**2 + original_edges_y**2 + self.epsilon)

        # 增强图像边缘
        enhanced_edges_x = F.conv2d(enhanced, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
        enhanced_edges_y = F.conv2d(enhanced, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
        # 关键修改：在 sqrt 内部加入 epsilon
        enhanced_edges = torch.sqrt(enhanced_edges_x**2 + enhanced_edges_y**2 + self.epsilon)

        edge_loss = F.l1_loss(enhanced_edges, original_edges)
        return edge_loss
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


# ==================== 6. train_epoch / validate / test （略） ====================
# ...... 与原文完全一致，此处省略 ......

# ==================== 7. 工具函数 ====================
def calculate_metrics(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    pa = np.diag(cm).sum() / cm.sum()
    cpa = np.diag(cm) / cm.sum(axis=1)
    mpa = np.nanmean(cpa)
    iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    mIoU = np.nanmean(iou)
    return pa, cpa, mpa, mIoU


def read_txt_to_list(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f'读取文件时出错: {e}')
        return []


def plot_training_history(train_history, save_dir):
    if not train_history:
        return
    epochs = [e['epoch'] for e in train_history]
    train_losses = [e['train_loss'] for e in train_history]
    val_mIoUs = [e['val_metrics']['mIoU'] for e in train_history]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(epochs, train_losses, 'b-', label='train loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid()
    ax2.plot(epochs, val_mIoUs, 'r-', label='val mIoU')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('mIoU'); ax2.legend(); ax2.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ 训练历史图表已保存')


def initialize_weights(model):
    for name, param in model.named_parameters():
        if param.dim() > 1 and 'weight' in name:
            nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        elif 'bias' in name:
            nn.init.constant_(param, 0)
    print('✅ 模型权重初始化完成')


# ==================== 8. 断点续训 ====================
def save_checkpoint(state, is_best, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    last_file = os.path.join(save_dir, 'last_checkpoint.pth')
    torch.save(state, last_file)
    if is_best:
        best_file = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_file)
        print(f'✅ 保存最佳模型 → {best_file}')

def load_checkpoint(model, save_dir, device, optimizer=None, scheduler=None, strict=True):
    """
    加载检查点（完全兼容 PyTorch 2.6+）
    由于 numpy dtype 不在白名单，强制 weights_only=False
    """
    last_file = os.path.join(save_dir, 'last_checkpoint.pth')
    if not os.path.exists(last_file):
        print('⚠️ 未发现检查点，从头开始训练。')
        return 0, 0.0, []

    print(f'✅ 发现检查点：正在加载 {last_file} ...')
    # 允许 numpy 相关全局对象（即使 weights_only=False 也 harmless）
    torch.serialization.add_safe_globals([np.dtype, np.core.multiarray.scalar,
                                          np.dtypes.Float64DType, np.dtypes.Float32DType])
    # 关键：PyTorch 2.6 默认 True，这里显式回退
    checkpoint = torch.load(last_file, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_mIoU = checkpoint.get('best_mIoU', 0.0)
    history = checkpoint.get('history', [])
    print(f'✅ 加载完成。将从 Epoch {start_epoch} 开始，当前最佳 mIoU: {best_mIoU:.4f}')
    return start_epoch, best_mIoU, history

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
        dsm = cv2.imread(self.depth_folder.format(id), cv2.IMREAD_ANYDEPTH)  # uint16
        if dsm is None:
            raise FileNotFoundError(self.depth_folder.format(id))
        dsm = cv2.resize(dsm, (int(dsm.shape[1] * scale), int(dsm.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)

        # 归一化到 0-1（用全局最大最小，可改成数据集统计值）
        d_min, d_max = dsm.min(), dsm.max()
        dsm = (dsm.astype(np.float32) - d_min) / (d_max - d_min + 1e-8)
        
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
        
        dsm_tensor = torch.tensor(np.array(dsm_patch_resized), dtype=torch.float32).unsqueeze(0) / 255.0
        label_tensor = torch.tensor(np.array(label_patch_resized), dtype=torch.long)

        # 【关键修复】最终安全检查
        if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
            # 如果发现坏数据，返回全0张量代替，防止训练崩溃
            print(f"警告: 发现坏数据 ID {id}")
            img_tensor = torch.zeros_like(img_tensor)
            
        if torch.isnan(dsm_tensor).any() or torch.isinf(dsm_tensor).any():
            dsm_tensor = torch.zeros_like(dsm_tensor)

        # 确保标签没有非法值 (假设 num_classes=16)
        # 将非法标签设为 255 (PyTorch通常用255作为忽略索引)
        label_tensor[label_tensor < 0] = 255
        label_tensor[label_tensor >= 16] = 255 
        
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
def check_nan_hook(module, input, output):
    """
    前向传播 Hook，用于检查模块输出中是否存在 NaN 或 Inf 值。
    已优化内存安全性，确保检查结果不保留计算图引用。
    """
    
    # 辅助函数：安全地检查张量中的 NaN/Inf
    def _safe_check_bad_values(t):
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            # 使用 .any().item() 确保检查结果立即转换为 Python 布尔值
            # 检查 NaN (Not a Number) 和 Inf (Infinity)
            has_nan = torch.isnan(t).any().item()
            has_inf = torch.isinf(t).any().item()
            return has_nan or has_inf
        return False

    # 1. 检查单个 Tensor 输出
    if isinstance(output, torch.Tensor):
        if _safe_check_bad_values(output):
            error_msg = f"❌ NaN or Inf detected in {module.__class__.__name__} output."
            print(error_msg)
            # 抛出运行时错误，立即中断训练
            raise RuntimeError(error_msg)
            
    # 2. 检查 tuple/list 形式的多个 Tensor 输出 (例如 Transformer Blocks 或 RNN)
    elif isinstance(output, (tuple, list)):
        for i, out in enumerate(output):
            if _safe_check_bad_values(out):
                error_msg = f"❌ NaN or Inf detected in {module.__class__.__name__} output[{i}]."
                print(error_msg)
                # 抛出运行时错误，立即中断训练
                raise RuntimeError(error_msg)
def test_model(model, test_loader, device, output_dir, logger=None):
    """测试模型，增加logger参数"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用logger或默认打印
    log_func = logger.log if logger else print
    
    all_preds = []
    all_gts = []
    
    # 【内存/显存优化】在测试开始前清理一次 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for i, (images, depths, labels) in enumerate(tqdm(test_loader, desc='测试中')):
            # 1. 数据转移到设备
            images = images.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            # 标签直接转到 CPU numpy，不会占用 GPU 显存，保持原样
            labels_cpu = labels.cpu().numpy()
            
            # 2. 前向传播
            enhanced_output, seg_output = model(images, depths)
            
            # 3. 获取预测结果
            _, preds_tensor = torch.max(seg_output, 1)
            
            # 【内存/显存优化】将 preds 转移到 CPU 并转为 numpy，避免 GPU 积累
            preds_numpy = preds_tensor.cpu().numpy()
            
            all_preds.extend(preds_numpy)
            all_gts.extend(labels_cpu)
            
            # 4. 可视化部分结果
            if i < 5:
                # 【内存/显存优化】
                # 将需要可视化张量转移到 CPU，避免 visualize_results 依赖 GPU 张量，
                # 并防止在函数调用期间保留对这些张量的引用。
                
                # 注意：我们应该使用原始 images 的 .cpu() 版本，而不是 GPU 上的引用
                # 如果 images 已经被原地修改，这里需要使用 .detach() 确保安全
                vis_images = images.detach().cpu()
                vis_enhanced = enhanced_output.detach().cpu()
                
                visualize_results(vis_images, vis_enhanced, preds_numpy, labels_cpu, output_dir, i)
                
                # 【内存/显存优化】显式删除临时变量
                del vis_images, vis_enhanced

            # 5. 【内存/显存优化】显式删除批次内的 GPU 张量，防止在循环中积累
            del images, depths, enhanced_output, seg_output, preds_tensor
            
            # 【内存/显存优化】每隔固定批次清理一次缓存，例如每 100 批次
            if i % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 【内存/显存优化】测试循环结束后，清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 计算指标
    # 注意：这里假设 calculate_metrics 函数能够处理扁平化的 numpy 数组
    try:
        pa, cpa, mpa, mIoU = calculate_metrics(
            np.concatenate([p.ravel() for p in all_preds]),
            np.concatenate([g.ravel() for g in all_gts])
        )
    except Exception as e:
        log_func(f"❌ 指标计算失败: {e}. 请检查 all_preds/all_gts 的数据格式。")
        # 返回默认的空结果
        pa, cpa, mpa, mIoU = 0.0, [], 0.0, 0.0
        
    
    # 保存结果
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"PA: {pa:.4f}\n")
        f.write(f"MPA: {mpa:.4f}\n")
        f.write(f"mIoU: {mIoU:.4f}\n")
        f.write(f"CPA: {cpa.tolist() if isinstance(cpa, np.ndarray) else cpa}\n")
    
    log_func(f"测试完成 - PA: {pa:.4f}, MPA: {mpa:.4f}, mIoU: {mIoU:.4f}")
    
    return all_preds, all_gts, {
        'pa': pa, 
        'mpa': mpa, 
        'mIoU': mIoU,
        # 确保 cpa 是一个可序列化的列表
        'cpa': cpa.tolist() if isinstance(cpa, np.ndarray) else cpa
    }
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
        axes[0, 0].set_title('origin RGB')
        axes[0, 0].axis('off')
        
        # 增强后图像
        enhanced_np = enhanced[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, 1].imshow(np.clip(enhanced_np, 0, 1))
        axes[0, 1].set_title('enhanced RGB')
        axes[0, 1].axis('off')
        
        # 预测结果
        axes[1, 0].imshow(predictions[i], cmap='tab20')
        axes[1, 0].set_title('pred')
        axes[1, 0].axis('off')
        
        # 真实标签
        axes[1, 1].imshow(labels[i], cmap='tab20')
        axes[1, 1].set_title('gt')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'result_batch{batch_idx}_sample{i}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close('all')

# ==================== 9. 主入口 ====================
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ---- 设备 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}  '
              f'({torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB)')

    # ---- 数据 ID ----
    train_ids = read_txt_to_list('train1.txt')
    val_ids = read_txt_to_list('val1.txt')
    test_ids = read_txt_to_list('test1.txt')

    DATA_FOLDER = '/root/autodl-tmp/dataset/cropped_Image/{}.jpg'
    DSM_FOLDER = '/root/autodl-tmp/dataset/cropped_Depth/{}.png'
    LABEL_FOLDER = '/root/autodl-tmp/dataset/cropped_Label/{}_label.png'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = EndToEndDataset(train_ids, DATA_FOLDER, DSM_FOLDER, LABEL_FOLDER, transform)
    val_dataset = EndToEndDataset(val_ids, DATA_FOLDER, DSM_FOLDER, LABEL_FOLDER, transform)
    test_dataset = EndToEndDataset(test_ids, DATA_FOLDER, DSM_FOLDER, LABEL_FOLDER, transform)

    BATCH_SIZE = 8
    NUM_WORKERS = 64 if device.type == 'cuda' else 4
    pin_mem = device.type == 'cuda'

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin_mem,
                              persistent_workers=pin_mem and NUM_WORKERS > 0,
                              prefetch_factor=2 if NUM_WORKERS > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin_mem,
                            persistent_workers=pin_mem and NUM_WORKERS > 0,
                            prefetch_factor=2 if NUM_WORKERS > 0 else None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin_mem,
                             persistent_workers=pin_mem and NUM_WORKERS > 0,
                             prefetch_factor=2 if NUM_WORKERS > 0 else None)

    # ---- 模型 ----
    print('\nInitializing model...')
    moe_model = MoE_ISP_Model(num_experts=3)
    initialize_weights(moe_model)
    moe_model = moe_model.to(device)

    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 16
    config_vit.n_skip = 3
    config_vit.patches.grid = (256 // 16, 256 // 16)

    ftransunet_model = ViT_seg(config_vit, img_size=256, num_classes=16)
    pretrained_path = getattr(config_vit, 'pretrained_path', None)
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            ftransunet_model.load_from(weights=np.load(pretrained_path, allow_pickle=True))
            print('✅ FTransUNet pretrained weights loaded')
        except Exception as e:
            print(f'⚠️  Failed to load pretrained FTransUNet: {e}')
    ftransunet_model = ftransunet_model.to(device)

    model = EndToEndModel(moe_model, ftransunet_model).to(device)

    # ---- 日志 ----
    save_dir = './checkpoints'
    logger = ExperimentLogger(save_dir)

    # ---- 断点续训（只恢复模型，不恢复 optimizer） ----
    start_epoch, best_mIoU, train_history = load_checkpoint(model, save_dir, device,
                                                            optimizer=None, scheduler=None)

    # ---- 注册 NaN hook ----
    for name, module in model.named_modules():
        module.register_forward_hook(check_nan_hook)

    # ---- 参数统计 ----
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    # ---- 保存实验配置 ----
    experiment_config = {
        "model": "MoE-Enhancer + FTransUNet",
        "input_size": 256,
        "num_classes": 16,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "optimizer": "AdamW",
        "learning_rate": "stage-wise",
        "total_epochs": 500,
        "start_epoch": start_epoch,
        "device": str(device),
        "experiment_id": logger.experiment_id
    }
    logger.save_config(experiment_config)

    # ---- 训练 ----
    total_epochs = 500
    logger.log(f'\nStart training for {total_epochs} epochs ...')
    train_history, best_mIoU = train_progressive(
        model, train_loader, val_loader,
        optimizer=None, scheduler=None,  # 内部会按阶段创建
        device=device, save_dir=save_dir,
        total_epochs=total_epochs,
        logger=logger,
        start_epoch=start_epoch,
        best_mIoU=best_mIoU,
        history=train_history
    )
    logger.log(f'\nTraining finished!  best mIoU: {best_mIoU:.4f}')

    # ---- 测试 ----
    logger.log('\nStart testing ...')
    best_model_file = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_file):
        # 仅加载权重
        torch.serialization.add_safe_globals([np.dtype, np.core.multiarray.scalar,
                                      np.dtypes.Float64DType, np.dtypes.Float32DType])
        ckpt = torch.load(best_model_file, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        logger.log(f'✅ Loaded best model (epoch {ckpt["epoch"]}, mIoU {ckpt["best_mIoU"]:.4f})')
    else:
        logger.log('⚠️  best_model.pth not found, use last checkpoint instead')

    test_output_dir = os.path.join(save_dir, 'test_results')
    all_preds, all_gts, test_metrics = test_model(model, test_loader, device, test_output_dir, logger)

    # ---- 保存最终结果 ----
    final_results = {
        "best_val_mIoU": best_mIoU,
        "test_metrics": test_metrics,
        "training_history": train_history
    }
    logger.save_results(final_results)

    # ---- 打印汇总 ----
    logger.log('\n' + '=' * 50)
    logger.log('Final summary')
    logger.log('=' * 50)
    logger.log(f'Test set size: {len(test_ids)}')
    logger.log(f'PA : {test_metrics["pa"]:.4f}')
    logger.log(f'MPA: {test_metrics["mpa"]:.4f}')
    logger.log(f'mIoU: {test_metrics["mIoU"]:.4f}')
    plot_training_history(train_history, save_dir)
    logger.log(f'\nAll done!  Results saved to {save_dir}')