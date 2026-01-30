# -*- coding: utf-8 -*-
"""
三阶段渐进训练：深度补全(RDFCGAN) + 图像增强 + 语义分割
（集成RDFCGAN对抗训练）
"""
# ==================== 导入 ====================
import os
import gc
import time
import numpy as np
from glob import glob
from tqdm import tqdm
import random
import itertools
import json
from datetime import datetime
import cv2
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.autograd as autograd
from torch.autograd import Variable
import torch.cuda.amp as amp
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

# 模型导入
from gan import build_model
from lib.utils.configurator import cfg, load_config
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from enhancement import MoE_ISP_Model, ImageQualityRegularizer
from utils import calculate_metrics, read_txt_to_list, plot_training_history, initialize_weights, check_nan_hook

# ==================== 1. RDFCGAN包装器 ====================
class RDFCGAN_Wrapper(nn.Module):
    """包装RDFCGAN以适配三阶段接口"""
    def __init__(self, rdgan_model):
        super().__init__()
        self.rdgan = rdgan_model
        
    def forward(self, rgb, raw_depth, mask_missing):
        """
        适配接口：输入rgb, raw_depth, mask_missing
        输出: {'pred_depth': ...}
        """
        # RDFCGAN的forward_test期望特定的输入格式
        batch_size = rgb.shape[0]
        
        # 准备输入字典（模拟RDFCGAN的set_input）
        input_dict = {
            'rgb': rgb,
            'gt_depth': torch.zeros(batch_size, 1, rgb.shape[2], rgb.shape[3]).to(rgb.device),  # 占位符
            'raw_depth': raw_depth.squeeze(1),  # RDFCGAN期望[H,W]格式
            'depth_masks': mask_missing.squeeze(1),  # RDFCGAN期望[H,W]格式
            'gt_normal': torch.zeros(batch_size, 3, rgb.shape[2], rgb.shape[3]).to(rgb.device),  # 占位符
            'normal_masks': torch.ones(batch_size, 1, rgb.shape[2], rgb.shape[3]).to(rgb.device),  # 占位符
            'labels': torch.zeros(batch_size, rgb.shape[2], rgb.shape[3]).long().to(rgb.device)  # 占位符
        }
        
        # 调用RDFCGAN的forward_test
        ret = self.rdgan.forward_test(
            rgb=rgb,
            raw_depth=raw_depth.squeeze(1),
            mask_missing=mask_missing.squeeze(1)
        )
        
        # 确保输出是4D张量 [B,1,H,W]
        pred_depth = ret['pred_depth']
        if pred_depth.dim() == 3:
            pred_depth = pred_depth.unsqueeze(1)
        
        return {'pred_depth': pred_depth}
    
    def train(self, mode=True):
        """重写train以适配RDFCGAN内部结构"""
        self.rdgan.train(mode)
        return self
    
    def eval(self):
        """重写eval"""
        self.rdgan.eval()
        return self

# ==================== 2. 端到端三阶段模型 ====================
class EndToEndDepthCompletionEnhancementSegmentation(nn.Module):
    def __init__(self, depth_completion_model, enhancement_model, segmentation_model,
                 freeze_depth=False, freeze_enh=False, freeze_seg=False):
        super().__init__()
        # 包装RDFCGAN
        if hasattr(depth_completion_model, 'forward_test'):  # 如果是原始RDFCGAN
            self.depth_completion_model = RDFCGAN_Wrapper(depth_completion_model)
        else:
            self.depth_completion_model = depth_completion_model
            
        self.enhancement_model = enhancement_model
        self.segmentation_model = segmentation_model

        # 冻结策略
        if freeze_depth:
            for p in self.depth_completion_model.parameters():
                p.requires_grad = False
        if freeze_enh:
            for p in self.enhancement_model.parameters():
                p.requires_grad = False
        if freeze_seg:
            for p in self.segmentation_model.parameters():
                p.requires_grad = False

    def forward(self, rgb, depth, mask_missing):
        """
        前向传播三阶段：深度补全→增强→语义分割
        """
        # ███ 核心修复1：统一转换为4D张量 [B,1,H,W] ███
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if mask_missing.dim() == 3:
            mask_missing = mask_missing.unsqueeze(1)
        
        # ========== 阶段1：深度补全 ==========
        ret = self.depth_completion_model(rgb, depth, mask_missing)
        pred_depth = ret['pred_depth']
        
        # 确保输出是4D [B,1,H,W]
        if pred_depth.dim() == 3:
            pred_depth = pred_depth.unsqueeze(1)
        if pred_depth.shape[1] != 1:
            pred_depth = pred_depth[:, 0:1, :, :]
        
        # 填充缺失区域
        completed_depth = depth * (1 - mask_missing) + pred_depth * mask_missing
        
        # ========== 阶段2：图像增强 ==========
        # MoE增强器需要 [B, H, W] 的深度输入
        depth_for_enh = completed_depth.squeeze(1)
        enhanced_rgb = self.enhancement_model(rgb, depth_for_enh)
        enhanced_rgb = torch.clamp(enhanced_rgb, 0.0, 1.0)
        
        # ========== 阶段3：语义分割 ==========
        # FTransUNet需要 [B, H, W] 的深度
        depth_for_seg = completed_depth.squeeze(1)
        seg_logits = self.segmentation_model(enhanced_rgb, depth_for_seg)
        
        return completed_depth, enhanced_rgb, seg_logits

# ==================== 3. 三任务损失函数 ====================
class TripleTaskLoss(nn.Module):
    def __init__(self, depth_weight=0.5, enh_weight=0.3, seg_weight=1.0, 
                 class_weights=None, enh_weights=None):
        super().__init__()
        self.depth_weight = depth_weight
        self.enh_weight = enh_weight
        self.seg_weight = seg_weight
        
        # 深度补全损失
        self.depth_criterion = nn.SmoothL1Loss(reduction='none')
        
        # 图像增强质量损失
        if enh_weights is None:
            enh_weights = {'ssim': 0.3, 'edge': 0.3, 'color': 0.2, 'exposure': 0.2}
        self.enh_criterion = ImageQualityRegularizer(enh_weights)
        
        # 语义分割损失
        self.seg_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        
    def forward(self, completed_depth, enhanced_rgb, seg_logits,
                target_depth, target_labels, mask_missing, original_rgb):
        """
        计算三个任务的损失
        """
        # ========== 深度补全损失 ==========
        valid_mask = 1.0 - mask_missing
        depth_diff = self.depth_criterion(completed_depth, target_depth)
        depth_loss = torch.sum(depth_diff * valid_mask) / (torch.sum(valid_mask) + 1e-8)
        
        # ========== 图像增强损失 ==========
        if valid_mask.dim() == 3:
            valid_mask_3ch = valid_mask.unsqueeze(1).expand_as(enhanced_rgb)
        else:
            valid_mask_3ch = valid_mask.expand_as(enhanced_rgb)
        
        enh_loss_total, enh_details = self.enh_criterion(
            enhanced_rgb * valid_mask_3ch, 
            original_rgb * valid_mask_3ch
        )
        
        # ========== 语义分割损失 ==========
        seg_loss = self.seg_criterion(seg_logits, target_labels)
        
        # ========== 总损失 ==========
        total_loss = (self.depth_weight * depth_loss + 
                      self.enh_weight * enh_loss_total + 
                      self.seg_weight * seg_loss)
        
        return total_loss, depth_loss, enh_loss_total, seg_loss, enh_details

# ==================== 4. 实验记录器 ====================
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

# ==================== 5. 工具函数 ====================
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
    """
    last_file = os.path.join(save_dir, 'last_checkpoint.pth')
    if not os.path.exists(last_file):
        print('⚠️ 未发现检查点，从头开始训练。')
        return 0, 0.0, []

    print(f'✅ 发现检查点：正在加载 {last_file} ...')
    torch.serialization.add_safe_globals([np.dtype, np.core.multiarray.scalar,
                                          np.dtypes.Float64DType, np.dtypes.Float32DType])
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

def calculate_metrics(y_pred, y_true):
    """计算分割指标"""
    cm = confusion_matrix(y_true, y_pred)
    pa = np.diag(cm).sum() / cm.sum()
    cpa = np.diag(cm) / cm.sum(axis=1)
    mpa = np.nanmean(cpa)
    iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    mIoU = np.nanmean(iou)
    return pa, cpa, mpa, mIoU

# ==================== 6. 训练、验证、测试 ====================
def train_epoch_triple(model, train_loader, optimizer, criterion, device, 
                       stage, logger=None):
    """训练一个epoch（集成RDFCGAN的对抗训练）"""
    model.train()
    log_func = logger.log if logger else print
    
    total_loss = 0.0
    depth_loss_total = 0.0
    enh_loss_total = 0.0
    seg_loss_total = 0.0
    g_loss_total = 0.0
    d_loss_total = 0.0
    
    enh_details_accum = {'ssim': 0.0, 'edge': 0.0, 'color': 0.0, 'exposure': 0.0}
    
    progress_bar = tqdm(train_loader, desc=f'阶段{stage} 训练中')
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for batch_idx, (images, depths, labels, masks) in enumerate(progress_bar):
        try:
            # 数据准备
            images = images.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # ███ RDFCGAN对抗训练部分 ███
            # 设置RDFCGAN的输入
            raw_depths = depths * (1 - masks)
            if hasattr(model.depth_completion_model, 'rdgan'):
                rdgan = model.depth_completion_model.rdgan
                # 准备RDFCGAN的batch数据
                batch_data = {
                    'rgb': images,
                    'gt_depth': depths,
                    #'raw_depth': depths * (1 - masks),  # 模拟损坏
                    'raw_depth': raw_depths.squeeze(1),  # 损坏的深度
                    'depth_mask': masks.squeeze(1),  # missing mask
                    #'depth_mask': masks,
                    'gt_normal': torch.zeros_like(images[:, :3]),  # 占位符
                    'normal_masks': torch.ones_like(masks),
                    'labels': labels
                }
                rdgan.set_input(batch_data)
                rdgan.forward()  # 前向传播
                rdgan.optimize_parameters()  # D/G交替优化
            
            # ███ 三阶段模型前向传播 ███
            optimizer.zero_grad()
            
            with amp.autocast():
                weights = model.parameters()
                completed_depth, enhanced_rgb, seg_logits = model(images.type_as(weights), raw_depths.type_as(weights), masks.type_as(weights))
                #completed_depth, enhanced_rgb, seg_logits = model(images, depths, masks)
                
                # 计算三任务损失
                loss, depth_loss, enh_loss, seg_loss, enh_details = criterion(
                    completed_depth, enhanced_rgb, seg_logits,
                    depths, labels, masks, images
                )
            
            # 检查NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                log_func(f"❌ 批次 {batch_idx}: 损失为 NaN/Inf，跳过")
                del images, depths, labels, masks
                del completed_depth, enhanced_rgb, seg_logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            
            # 反向传播（仅三阶段模型，RDFCGAN已单独优化）
            scaler = amp.GradScaler()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # 记录损失
            total_loss += loss.item()
            depth_loss_total += depth_loss.item()
            enh_loss_total += enh_loss.item()
            seg_loss_total += seg_loss.item()
            
            if hasattr(model.depth_completion_model, 'rdgan'):
                # 记录GAN损失
                g_loss_total += model.depth_completion_model.rdgan.loss_G.item()
                d_loss_total += model.depth_completion_model.rdgan.loss_D.item()
            
            for key in enh_details:
                enh_details_accum[key] += enh_details[key].item()
            
            if batch_idx % 10 == 0:
                postfix = {
                    'Total': f'{loss.item():.4f}',
                    'Depth': f'{depth_loss.item():.4f}',
                    'Enh': f'{enh_loss.item():.4f}',
                    'Seg': f'{seg_loss.item():.4f}'
                }
                if hasattr(model.depth_completion_model, 'rdgan'):
                    postfix['G_GAN'] = f'{model.depth_completion_model.rdgan.loss_G.item():.4f}'
                    postfix['D'] = f'{model.depth_completion_model.rdgan.loss_D.item():.4f}'
                progress_bar.set_postfix(postfix)
        
        except RuntimeError as e:
            log_func(f"❌ 运行时错误: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        
        finally:
            # 清理
            if 'images' in locals(): del images
            if 'depths' in locals(): del depths
            if 'labels' in locals(): del labels
            if 'masks' in locals(): del masks
            if 'completed_depth' in locals(): del completed_depth
            if 'enhanced_rgb' in locals(): del enhanced_rgb
            if 'seg_logits' in locals(): del seg_logits
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 计算平均
    num_batches = len(train_loader)
    avg_total = total_loss / num_batches
    avg_depth = depth_loss_total / num_batches
    avg_enh = enh_loss_total / num_batches
    avg_seg = seg_loss_total / num_batches
    avg_details = {k: v / num_batches for k, v in enh_details_accum.items()}
    
    # GAN损失
    if hasattr(model.depth_completion_model, 'rdgan'):
        avg_g_loss = g_loss_total / num_batches
        avg_d_loss = d_loss_total / num_batches
        avg_details['GAN_G'] = avg_g_loss
        avg_details['GAN_D'] = avg_d_loss
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return avg_total, {
        'depth_loss': avg_depth,
        'enh_loss': avg_enh,
        'seg_loss': avg_seg,
        'enh_details': avg_details
    }

def validate_model(model, val_loader, criterion, device, logger=None):
    """验证模型（使用RDFCGAN的forward_test）"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_gts = []
    
    log_func = logger.log if logger else print
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for images, depths, labels, masks in tqdm(val_loader, desc='验证中'):
            images = images.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # ███ 使用RDFCGAN的forward_test ███
            completed_depth, enhanced_rgb, seg_logits = model(images, depths, masks)
            
            # 计算分割损失
            seg_loss = criterion.seg_criterion(seg_logits, labels)
            total_loss += seg_loss.item()
            
            # 预测
            _, preds = torch.max(seg_logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())
            
            # 清理
            del images, depths, labels, masks
            del completed_depth, enhanced_rgb, seg_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 计算指标
    if all_preds and all_gts:
        pa, cpa, mpa, mIoU = calculate_metrics(
            np.concatenate([p.ravel() for p in all_preds]),
            np.concatenate([g.ravel() for g in all_gts])
        )
    else:
        pa, cpa, mpa, mIoU = 0, [], 0, 0
    
    log_func(f"验证指标 - PA: {pa:.4f}, MPA: {mpa:.4f}, mIoU: {mIoU:.4f}")
    
    return {
        'val_loss': total_loss / len(val_loader),
        'pa': pa,
        'mpa': mpa,
        'mIoU': mIoU,
        'cpa': cpa.tolist() if isinstance(cpa, np.ndarray) else cpa
    }

def test_triple_model(model, test_loader, device, output_dir, logger=None):
    """测试三阶段模型（集成RDFCGAN）"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    log_func = logger.log if logger else print
    all_preds = []
    all_gts = []
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for i, (images, depths, labels, masks) in enumerate(tqdm(test_loader, desc='测试')):
            images = images.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels_cpu = labels.numpy()
            
            # ███ 使用RDFCGAN的forward_test ███
            completed_depth, enhanced_rgb, seg_logits = model(images, depths, masks)
            
            _, preds = torch.max(seg_logits, 1)
            preds_numpy = preds.cpu().numpy()
            
            all_preds.extend(preds_numpy)
            all_gts.extend(labels_cpu)
            
            # 可视化
            if i < 5:
                vis_completed = completed_depth.cpu()
                vis_enhanced = enhanced_rgb.cpu()
                vis_images = images.cpu()
                
                for j in range(min(3, images.shape[0])):
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # 原始RGB
                    img_np = vis_images[j].numpy().transpose(1, 2, 0)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = std * img_np + mean
                    axes[0, 0].imshow(np.clip(img_np, 0, 1))
                    axes[0, 0].set_title('Original RGB')
                    
                    # 增强RGB
                    enh_np = vis_enhanced[j].numpy().transpose(1, 2, 0)
                    axes[0, 1].imshow(np.clip(enh_np, 0, 1))
                    axes[0, 1].set_title('Enhanced RGB')
                    
                    # 补全深度
                    depth_np = vis_completed[j].squeeze().numpy()
                    im = axes[1, 0].imshow(depth_np, cmap='magma')
                    axes[1, 0].set_title('Completed Depth')
                    plt.colorbar(im, ax=axes[1, 0])
                    
                    # 预测结果
                    axes[1, 1].imshow(preds_numpy[j], cmap='tab20')
                    axes[1, 1].set_title('Prediction')
                    
                    for ax in axes.flat:
                        ax.axis('off')
                    
                    plt.tight_layout()
                    save_path = os.path.join(output_dir, f'test_sample_{i}_{j}.png')
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
            
            # 清理
            del images, depths, labels, masks
            del completed_depth, enhanced_rgb, seg_logits
            if i % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 计算指标
    if all_preds and all_gts:
        pa, cpa, mpa, mIoU = calculate_metrics(
            np.concatenate([p.ravel() for p in all_preds]),
            np.concatenate([g.ravel() for g in all_gts])
        )
    else:
        pa, cpa, mpa, mIoU = 0, [], 0, 0
    
    results = {
        'pa': pa, 'mpa': mpa, 'mIoU': mIoU,
        'cpa': cpa.tolist() if isinstance(cpa, np.ndarray) else cpa
    }
    
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    log_func(f'测试完成 - PA: {pa:.4f}, MPA: {mpa:.4f}, mIoU: {mIoU:.4f}')
    return results
    
def train_progressive_triple(model, train_loader, val_loader, device, save_dir, 
                             total_epochs=500, logger=None, start_epoch=0, 
                             best_mIoU=0.0, history=None):
    """
    三阶段渐进式训练：
    阶段1：仅训练深度补全（10轮预热）
    阶段2：深度补全 + 分割联合训练
    阶段3：三任务联合训练
    """
    os.makedirs(save_dir, exist_ok=True)
    if logger is None:
        logger = ExperimentLogger(save_dir)
    history = history or []

    stage_configs = [
        {'name': '阶段1-深度补全预热', 'epochs': 1, 'stage': 1,
         'depth_w': 1.0, 'enh_w': 0.0, 'seg_w': 0.0, 'lr': 1e-4},
        {'name': '阶段2-深度+分割联合', 'epochs': 1, 'stage': 2,
         'depth_w': 0.5, 'enh_w': 0.0, 'seg_w': 1.0, 'lr': 1e-5},
        {'name': '阶段3-三任务平衡', 'epochs': 1, 'stage': 3,
         'depth_w': 0.3, 'enh_w': 0.2, 'seg_w': 1.0, 'lr': 1e-6}
    ]

    epochs_done = 0
    for cfg in stage_configs:
        name, epochs, stage = cfg['name'], cfg['epochs'], cfg['stage']
        depth_w, enh_w, seg_w, lr = cfg['depth_w'], cfg['enh_w'], cfg['seg_w'], cfg['lr']
        
        if start_epoch >= epochs_done + epochs:
            epochs_done += epochs
            logger.log(f'跳过 {name}，已完成 {epochs} 轮')
            continue
        
        logger.log(f'\n{"="*60}\n开始 {name}\n{"="*60}')
        
        # 创建阶段优化器和损失
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        criterion = TripleTaskLoss(depth_weight=depth_w, enh_weight=enh_w, seg_weight=seg_w).to(device)
        
        # 参数冻结策略
        if stage == 1:
            # 仅训练深度补全
            for p in model.enhancement_model.parameters():
                p.requires_grad = False
            for p in model.segmentation_model.parameters():
                p.requires_grad = False
        elif stage == 2:
            # 解冻深度和分割
            for p in model.depth_completion_model.parameters():
                p.requires_grad = True
            for p in model.segmentation_model.parameters():
                p.requires_grad = True
            for p in model.enhancement_model.parameters():
                p.requires_grad = False
        else:
            # 三任务全解冻
            for p in model.parameters():
                p.requires_grad = True
        
        for epoch in range(epochs):
            curr_epoch = epochs_done + epoch + 1
            if curr_epoch < start_epoch:
                continue
            
            # 训练
            train_loss, train_details = train_epoch_triple(
                model, train_loader, optimizer, criterion, device, stage, logger
            )
            
            # 验证
            if curr_epoch % 1 == 0:
                val_metrics = validate_model(model, val_loader, criterion, device, logger)
                scheduler.step(val_metrics['val_loss'])
                
                is_best = val_metrics['mIoU'] > best_mIoU
                best_mIoU = max(best_mIoU, val_metrics['mIoU'])
                
                # 保存checkpoint
                save_checkpoint({
                    'epoch': curr_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_mIoU': best_mIoU,
                    'stage': stage
                }, is_best, save_dir)
                
                logger.log(f'Epoch {curr_epoch}: TL={train_loss:.4f}, '
                           f'DL={train_details["depth_loss"]:.4f}, '
                           f'EL={train_details["enh_loss"]:.4f}, '
                           f'SL={train_details["seg_loss"]:.4f}, '
                           f'Val mIoU={val_metrics["mIoU"]:.4f}')
            
            # 周期性清理
            if curr_epoch % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        epochs_done += epochs
        del criterion, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return history, best_mIoU
    
class DepthCompletionDataset(torch.utils.data.Dataset):
    """深度补全数据集：RGB + 深度图 + 标签 + 模拟损坏"""
    def __init__(self, ids, rgb_folder, depth_folder, label_folder=None, target_size=(256, 256), 
                 missing_rate=0.2):
        self.ids = ids
        self.rgb_fmt = rgb_folder
        self.depth_fmt = depth_folder
        self.label_fmt = label_folder  # 新增：标签路径格式
        self.target_size = target_size
        self.missing_rate = missing_rate
        
        # RGB归一化
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        
        # RGB图像
        rgb = Image.open(self.rgb_fmt.format(id_)).convert('RGB')
        rgb = rgb.resize(self.target_size, Image.BILINEAR)
        
        # 深度图（16-bit）
        depth16 = cv2.imread(self.depth_fmt.format(id_), cv2.IMREAD_ANYDEPTH)
        depth16 = cv2.resize(depth16, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # 归一化到[0,1]
        d_min, d_max = depth16.min(), depth16.max()
        depth_arr = (depth16.astype(np.float32) - d_min) / (d_max - d_min + 1e-8)
        
        rgb_tensor = self.rgb_transform(rgb)
        depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0)  # [1, H, W]
        
        # 加载标签（如果提供了label_fmt）
        if self.label_fmt is not None and self.label_fmt != "":
            label = Image.open(self.label_fmt.format(id_)).convert('L')
            label = label.resize(self.target_size, Image.NEAREST)
            label_tensor = torch.from_numpy(np.array(label)).long()
        else:
            label_tensor = torch.zeros(self.target_size, dtype=torch.long)
        
        # 模拟损坏：随机丢失像素
        # valid_mask: 1 for valid, 0 for missing
        valid_mask = torch.rand_like(depth_tensor) > self.missing_rate
        # missing_mask: 1 for missing, 0 for valid (for model input)
        missing_mask = (~valid_mask).float()
        
        # 返回元组：images, depths, labels, masks
        # depths是ground truth depth，masks是missing mask (1=缺失)
        # 在训练循环中，损坏的深度 = depths * (1 - masks)
        return rgb_tensor, depth_tensor, label_tensor, missing_mask
        
def read_txt(file_path):
    """读取文本文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]
# ==================== 7. 主入口 ====================
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据路径
    DATA_ROOT = '/root/autodl-tmp/dataset'
    rgb_fmt = os.path.join(DATA_ROOT, 'Image', '{}.jpg')
    depth_fmt = os.path.join(DATA_ROOT, 'Depth', '{}.png')
    label_fmt = os.path.join(DATA_ROOT, 'Label_Label', '{}_label.png')

    # 读取数据ID
    train_ids = read_txt_to_list('train.txt')
    val_ids = read_txt_to_list('val.txt')
    test_ids = read_txt_to_list('test.txt')

    # 数据集
    train_set = DepthCompletionDataset(train_ids, rgb_fmt, depth_fmt, label_fmt, target_size=(256, 256))
    val_set = DepthCompletionDataset(val_ids, rgb_fmt, depth_fmt, label_fmt, target_size=(256, 256))
    test_set = DepthCompletionDataset(test_ids, rgb_fmt, depth_fmt, label_fmt, target_size=(256, 256))

    BATCH_SIZE = 8
    NUM_WORKERS = 4

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # 构建模型
    cfg_path = 'rdf_cycle_patchgan_config.yaml'
    load_config(cfg, cfg_path)
    
    # 1. RDFCGAN深度补全
    rdgan_model = build_model(cfg, device)
    
    # 2. MoE图像增强
    enh_model = MoE_ISP_Model(num_experts=3).to(device)
    initialize_weights(enh_model)
    
    # 3. FTransUNet语义分割
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 16
    config_vit.n_skip = 3
    config_vit.patches.grid = (256 // 16, 256 // 16)
    seg_model = ViT_seg(config_vit, img_size=256, num_classes=16).to(device)
    
    pretrained_path = getattr(config_vit, 'pretrained_path', None)
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            seg_model.load_from(np.load(pretrained_path, allow_pickle=True))
            print('✅ FTransUNet pretrained weights loaded')
        except Exception as e:
            print(f'⚠️ Failed to load pretrained FTransUNet: {e}')

    # 端到端模型
    model = EndToEndDepthCompletionEnhancementSegmentation(
        depth_completion_model=rdgan_model,
        enhancement_model=enh_model,
        segmentation_model=seg_model
    ).to(device)

    # 日志记录
    save_dir = './checkpoints_triple_stage_rdgan'
    logger = ExperimentLogger(save_dir)

    # 断点续训
    start_epoch, best_mIoU, history = load_checkpoint(
        model, save_dir, device, optimizer=None, scheduler=None
    )

    # 注册NaN检测
    for name, module in model.named_modules():
        module.register_forward_hook(check_nan_hook)

    # 训练配置
    experiment_config = {
        "model": "RDFCGAN+Enhancement+Segmentation",
        "num_classes": 16,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "total_epochs": 500,
        "start_epoch": start_epoch,
        "device": str(device),
        "stage_config": "渐进式三阶段训练（集成RDFCGAN对抗训练）"
    }
    logger.save_config(experiment_config)

    # 训练
    total_epochs = 500
    logger.log(f'\n开始 {total_epochs} 轮训练...')
    
    history, best_mIoU = train_progressive_triple(
        model, train_loader, val_loader, device, save_dir,
        total_epochs=total_epochs, logger=logger,
        start_epoch=start_epoch, best_mIoU=best_mIoU, history=history
    )
    
    logger.log(f'训练完成！最佳mIoU: {best_mIoU:.4f}')

    # 测试
    logger.log('\n开始测试...')
    best_model_file = os.path.join(save_dir, 'best_model.pth')
    
    if os.path.exists(best_model_file):
        torch.serialization.add_safe_globals([np.dtype, np.core.multiarray.scalar,
                                      np.dtypes.Float64DType, np.dtypes.Float32DType])
        ckpt = torch.load(best_model_file, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        logger.log(f'✅ 加载最佳模型 (epoch {ckpt["epoch"]}, mIoU {ckpt["best_mIoU"]:.4f})')
    
    test_output_dir = os.path.join(save_dir, 'test_results')
    test_metrics = test_triple_model(model, test_loader, device, test_output_dir, logger)

    # 保存结果
    final_results = {
        "best_val_mIoU": best_mIoU,
        "test_metrics": test_metrics,
        "training_history": history
    }
    logger.save_results(final_results)

    # 可视化
    plot_training_history(history, save_dir)
    logger.log(f'\n全部完成！结果保存至 {save_dir}')