#!/usr/bin/env python
# train_end2end_16bit.py
"""
端到端深度图补全 + 语义分割（16-bit 深度版）
"""
import os
import gc
import time
import numpy as np
from glob import glob
from tqdm import tqdm
import random
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.autograd as autograd
from torch.autograd import Variable
import torch.cuda.amp as amp
import torchvision.transforms as transforms

import cv2          
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

# ==================== 模型导入 ====================
from gan import build_model
from lib.utils.configurator import cfg, load_config
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def debug_dataset_and_mask(dataset, num_samples=3):
    """
    诊断深度图读取、归一化和掩码逻辑
    """
    for i in range(num_samples):
        print(f"\n--- 正在检查第 {i+1} 张样本 ---")
        rgb, depth, label, mask_missing = dataset[i]
        
        # 1. 检查原始深度信息
        # 注意：这里假设 dataset 返回的是 Tensor [1, H, W]
        d_np = depth.squeeze().numpy()
        m_np = mask_missing.squeeze().numpy()
        
        d_max = d_np.max()
        d_min = d_np.min()
        d_mean = d_np.mean()
        
        # 2. 检查 Mask 逻辑
        # mask_missing 为 1 的地方应该是深度为 0 的地方
        # valid_mask 为 1 的地方应该是有深度值的地方
        valid_mask = 1.0 - m_np
        num_valid = np.sum(valid_mask)
        total_pixels = d_np.size
        
        print(f"深度图 Tensor 统计: Max={d_max:.6f}, Min={d_min:.6f}, Mean={d_mean:.6f}")
        print(f"有效像素占比: {num_valid}/{total_pixels} ({num_valid/total_pixels:.2%})")
        
        # 3. 核心逻辑校验：计算 Masked Target Sum
        # 如果这个值为 0，说明有效区域内没有任何深度值，Loss 必然恒等于 0
        masked_target_sum = np.sum(d_np * valid_mask)
        print(f"Masked 深度之和 (Target Sum): {masked_target_sum:.6f}")
        
        if masked_target_sum == 0:
            print("❌ 错误确认：掩码过滤后的深度值为 0！Loss 恒等于 0 是因为 Mask 盖住了所有有效像素。")
        else:
            print("✅ 逻辑正常：掩码区域内存在有效深度值。")

        # 4. 可视化检查
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Normalized Depth")
        plt.imshow(d_np, cmap='magma')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.title("Valid Mask (1=Valid)")
        plt.imshow(valid_mask, cmap='gray')
        
        plt.subplot(1, 3, 3)
        plt.title("Masked Depth (Target)")
        plt.imshow(d_np * valid_mask, cmap='magma')
        plt.savefig("1.png")

# 使用示例（请替换为你自己的 dataset 实例）

class EndToEndTrainer:
    def __init__(self, model, train_loader, val_loader, device,
                 save_dir='./checkpoints_end2end_new', resume_path=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.start_epoch = 1
        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2)
        self.scaler = amp.GradScaler()
        self.criterion = MultiTaskLoss(seg_weight=1.0,
                               depth_weight=0.5,
                               grad_weight=0.05,   # ∇ 一致性权重
                               mid_weight=0.03)
        self.best_mIoU = 0.0

        if resume_path and os.path.exists(resume_path):
            self._resume_checkpoint(resume_path)

    def _resume_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_mIoU = ckpt.get('best_mIoU', 0.0)
        print(f'✅ 恢复训练 epoch {self.start_epoch}，best mIoU {self.best_mIoU:.4f}')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = depth_loss_total = seg_loss_total = 0.0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} 训练')
        for rgb, depth, labels, mask_missing in pbar:
            rgb, depth, labels, mask_missing = rgb.to(self.device), depth.to(self.device), labels.to(self.device), mask_missing.to(self.device)
            self.optimizer.zero_grad()
            with amp.autocast():
                # 现在模型返回三个值
                pred_raw, pred_comp, seg_logits = self.model(rgb, depth, mask_missing)
                
                # 传入原始预测值 pred_raw 进行深度监督
                loss, d_loss, s_loss = self.criterion(
                    pred_raw, pred_comp, seg_logits, 
                    depth, labels, mask_missing
                )
            # 在反向传播前添加
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            depth_loss_total += depth_loss.item()
            seg_loss_total += seg_loss.item()
            if pbar.n % 10 == 0:
                pbar.set_postfix({'Total': f'{loss.item():.4f}',
                                  'Depth': f'{depth_loss.item():.4f}',
                                  'Seg': f'{seg_loss.item():.4f}'})
        self.scheduler.step()
        return total_loss / len(self.train_loader), \
               depth_loss_total / len(self.train_loader), \
               seg_loss_total / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for rgb, depth, labels, mask_missing in tqdm(self.val_loader, desc='验证'):
                rgb,depth,labels,mask_missing=rgb.to(self.device),depth.to(self.device),labels.to(self.device),mask_missing.to(self.device)
                pred_depth, seg = self.model(rgb.type_as(next(self.model.parameters())), 
                                             depth.type_as(next(self.model.parameters())), 
                                             mask_missing.type_as(next(self.model.parameters())))
                all_preds.append(torch.argmax(seg, 1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        # 计算 mIoU
        y_pred = np.concatenate([p.ravel() for p in all_preds])
        y_true = np.concatenate([l.ravel() for l in all_labels])
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(16))
        iou = np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm) + 1e-8)
        mIoU = np.nanmean(iou)
        return mIoU

    def save_checkpoint(self, epoch, is_best):
        ckpt = {'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_mIoU': self.best_mIoU}
        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(ckpt, path)
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(ckpt, best_path)
            print(f'✅ 保存最佳模型 {best_path}')

    def train(self, epochs):
        print(f'训练 {epochs} 轮，从 epoch {self.start_epoch} 开始')
        for epoch in range(self.start_epoch, epochs + 1):
            train_loss, train_d, train_s = self.train_epoch(epoch)
            if epoch % 5 == 0:
                mIoU = self.validate(epoch)
                print(f'Epoch {epoch}: loss={train_loss:.4f}  '
                      f'depth_loss={train_d:.4f}  seg_loss={train_s:.4f}  mIoU={mIoU:.4f}')
                self.save_checkpoint(epoch, mIoU > self.best_mIoU)
                self.best_mIoU = max(self.best_mIoU, mIoU)
            else:
                self.save_checkpoint(epoch, False)

class FineTuneTrainer(EndToEndTrainer):
    """
    支持微调的训练器，提供更灵活的训练策略
    """
    def __init__(self, model, train_loader, val_loader, device,
                 save_dir='./checkpoints_end2end_new', resume_path=None,
                 finetune_lr=1e-6, freeze_depth_epochs=0):
        super().__init__(model, train_loader, val_loader, device, save_dir, resume_path)
        
        self.finetune_lr = finetune_lr
        self.freeze_depth_epochs = freeze_depth_epochs
        self.current_epoch = 0
        
        # 重新设置优化器以适应微调
        self._setup_finetune_optimizer()
    
    def _setup_finetune_optimizer(self):
        """设置微调优化器"""
        # 分离参数：深度补全参数和分割参数
        depth_params = []
        seg_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'depth_completion_model' in name:
                depth_params.append(param)
            elif 'segmentation_model' in name:
                seg_params.append(param)
            else:
                other_params.append(param)
        
        # 微调阶段使用不同的学习率
        self.optimizer = optim.AdamW([
            {'params': depth_params, 'lr': self.finetune_lr, 'weight_decay': 1e-4},
            {'params': seg_params, 'lr': self.finetune_lr * 10, 'weight_decay': 1e-4},  # 分割模型学习率稍大
            {'params': other_params, 'lr': self.finetune_lr, 'weight_decay': 1e-4}
        ])
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2)
    
    def train_epoch(self, epoch):
        self.current_epoch = epoch
        
        # 前N个epoch冻结深度补全模型
        if epoch <= self.freeze_depth_epochs:
            self._freeze_depth_model()
        else:
            self._unfreeze_depth_model()
        
        return super().train_epoch(epoch)
    
    def _freeze_depth_model(self):
        """冻结深度补全模型"""
        for name, param in self.model.named_parameters():
            if 'depth_completion_model' in name:
                param.requires_grad = False
    
    def _unfreeze_depth_model(self):
        """解冻深度补全模型"""
        for name, param in self.model.named_parameters():
            if 'depth_completion_model' in name:
                param.requires_grad = True

def staged_training():
    """
    分阶段训练：预训练 → 冻结微调 → 联合微调
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # 第一阶段：深度补全预训练
    print("=" * 50)
    print("🎯 第一阶段：深度补全模型预训练")
    print("=" * 50)

    DATA_ROOT = '/root/autodl-tmp/dataset'
    rgb_fmt   = os.path.join(DATA_ROOT, 'Image', '{}.jpg')
    depth_fmt = os.path.join(DATA_ROOT, 'Depth', '{}.png')
    label_fmt = os.path.join(DATA_ROOT, 'Label_Label', '{}_label.png')

    train_ids = read_txt('train.txt')
    val_ids   = read_txt('val.txt')

    train_set = EndToEndDataset(train_ids, rgb_fmt, depth_fmt, label_fmt,
                                target_size=(256, 256))
    #debug_dataset_and_mask(train_set)
    val_set   = EndToEndDataset(val_ids, rgb_fmt, depth_fmt, label_fmt,
                                target_size=(256, 256))

    train_loader = data.DataLoader(train_set, batch_size=8, shuffle=True,
                                   num_workers=4, pin_memory=True)
    val_loader   = data.DataLoader(val_set, batch_size=8, shuffle=False,
                                   num_workers=4, pin_memory=True)

    # 构建模型
    cfg_path = 'rdf_cycle_patchgan_config.yaml'
    load_config(cfg, cfg_path)
    depth_model = build_model(cfg, device)

    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 16
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(256 / 16), int(256 / 16))
    seg_model = ViT_seg(config_vit, img_size=256, num_classes=16).to(device)
    if os.path.exists(config_vit.pretrained_path):
        seg_model.load_from(np.load(config_vit.pretrained_path, allow_pickle=True))

    model = EndToEndDepthCompletionSegmentation(depth_model, seg_model)
    
    # 预训练深度补全模型
    depth_model = pretrain_depth_completion(
        depth_model, train_loader, val_loader, device,
        pretrain_epochs=200, save_dir='./checkpoints_end2end_new'
    )
    
    # 第二阶段：冻结深度补全，训练分割模型
    print("=" * 50)
    print("🎯 第二阶段：冻结深度补全，训练分割模型")
    print("=" * 50)
    
    model = EndToEndDepthCompletionSegmentation(
        depth_completion_model=depth_model,
        segmentation_model=seg_model,
        freeze_depth_model=True  # 冻结深度补全
    )
    
    # 使用微调训练器，前50个epoch冻结深度补全
    trainer = FineTuneTrainer(
        model, train_loader, val_loader, device,
        save_dir='./checkpoints_end2end_new',
        finetune_lr=1e-5,
        freeze_depth_epochs=50
    )
    trainer.train(epochs=150)
    
    # 第三阶段：联合微调
    print("=" * 50)
    print("🎯 第三阶段：联合微调")
    print("=" * 50)
    
    # 解冻深度补全模型
    for param in model.depth_completion_model.parameters():
        param.requires_grad = True
    
    # 使用更小的学习率进行联合微调
    trainer = FineTuneTrainer(
        model, train_loader, val_loader, device,
        save_dir='./checkpoints_end2end_new',
        finetune_lr=1e-6,
        freeze_depth_epochs=0  # 不冻结
    )
    trainer.train(epochs=150)

def pretrain_depth_completion(depth_model, train_loader, val_loader, device, 
                            pretrain_epochs=100, save_dir='./checkpoints_end2end_new'):
    """
    预训练深度补全模型
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 预训练优化器（使用较大学习率）
    optimizer = optim.AdamW(depth_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    scaler = amp.GradScaler()
    
    # 深度补全损失函数
    criterion = nn.SmoothL1Loss(beta=0.1)
    
    best_loss = float('inf')
    
    print("🚀 开始深度补全模型预训练...")
    for epoch in range(1, pretrain_epochs + 1):
        # 训练阶段
        depth_model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'预训练 Epoch {epoch}')
        for rgb, depth, _, mask_missing in pbar:  # 忽略标签
            rgb, depth, mask_missing = rgb.to(device), depth.to(device), mask_missing.to(device)
            
            optimizer.zero_grad()
            with amp.autocast():
                ret = depth_model.forward_test(rgb=rgb.type_as(next(depth_model.parameters())),
                                               raw_depth=depth.squeeze(1).type_as(next(depth_model.parameters())),
                                               mask_missing=mask_missing.type_as(next(depth_model.parameters())))
                pred_depth = ret['pred_depth']
                if pred_depth.shape[1] != 1:
                    pred_depth = pred_depth[:, 0:1, :, :]
                
                # 改进的 Masked Loss
                # 1. 我们只在【有效深度区域】计算 Loss，确保模型学会还原深度
                valid_mask = 1.0 - mask_missing
                
                # 如果有效像素太少，跳过该 batch 防止梯度爆炸
                if valid_mask.sum() < 10:
                    continue
            
                # 只计算非零区域的 L1 损失
                loss = F.smooth_l1_loss(pred_depth * valid_mask, depth * valid_mask, reduction='sum') 
                loss = loss / (valid_mask.sum() + 1e-8)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # 验证阶段
        if epoch % 10 == 0:
            val_loss = validate_depth_completion(depth_model, val_loader, criterion, device)
            print(f'Epoch {epoch}: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}')
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': depth_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss
                }, os.path.join(save_dir, 'best_depth_model.pth'))
                print(f'✅ 保存最佳深度补全模型，验证损失: {best_loss:.4f}')
    
    print(f"🎉 深度补全预训练完成！最佳验证损失: {best_loss:.4f}")
    return depth_model

def validate_depth_completion(model, val_loader, criterion, device):
    """验证深度补全模型"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for rgb, depth, _, mask_missing in val_loader:
            rgb, depth, mask_missing = rgb.to(device), depth.to(device), mask_missing.to(device)
            
            ret = model.forward_test(rgb=rgb.type_as(next(model.parameters())), 
                                     raw_depth=depth.squeeze(1).type_as(next(model.parameters())), 
                                     mask_missing=mask_missing.type_as(next(model.parameters())))
            pred_depth = ret['pred_depth']
            
            loss = criterion(pred_depth, depth)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

# 专用的深度补全数据集（不加载标签）
class DepthCompletionDataset(torch.utils.data.Dataset):
    def __init__(self, ids, rgb_folder, depth_folder, target_size=(256, 256)):
        self.ids = ids
        self.rgb_fmt = rgb_folder
        self.depth_fmt = depth_folder
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        
        # RGB图像
        rgb = Image.open(self.rgb_fmt.format(id_)).convert('RGB')
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=1))
        rgb = rgb.resize(self.target_size, Image.BICUBIC)
        
        # 深度图
        depth16 = cv2.imread(self.depth_fmt.format(id_), cv2.IMREAD_ANYDEPTH)
        depth16 = cv2.resize(depth16, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # 归一化
        d_min, d_max = depth16.min(), depth16.max()
        depth_arr = (depth16.astype(np.float32) - d_min) / (d_max - d_min + 1e-8)
        
        rgb_tensor = self.transform(rgb)
        depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0)
        
        return rgb_tensor, depth_tensor
        
# ==================== 端到端模型 ====================
class EndToEndDepthCompletionSegmentation(nn.Module):
    def __init__(self, depth_completion_model, segmentation_model,
                 freeze_depth_model=False, freeze_seg_model=False):
        super().__init__()
        self.depth_completion_model = depth_completion_model
        self.segmentation_model = segmentation_model

        if freeze_depth_model:
            for p in self.depth_completion_model.parameters():
                p.requires_grad = False
        if freeze_seg_model:
            for p in self.segmentation_model.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)  # 调用父类的 train 方法
        self.depth_completion_model.train(mode)  # 确保子模块也进入训练模式
        self.segmentation_model.train(mode)
    def eval(self):
        super().eval()
        self.depth_completion_model.eval()  # 确保子模块也进入训练模式
        self.segmentation_model.eval()
    
    def forward(self, rgb, depth, mask_missing):
        # 执行深度补全
        depth = depth.to(dtype=rgb.dtype, device=rgb.device)
        mask_missing = mask_missing.to(dtype=rgb.dtype, device=rgb.device)
        
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)  # [B,1,H,W]
        
        raw_depth = depth.squeeze(1)  # [B,H,W]
        
        # 深度补全前向传播
        ret = self.depth_completion_model.forward_test(
            rgb=rgb, 
            raw_depth=raw_depth, 
            mask_missing=mask_missing
        )
        
        pred_depth = ret['pred_depth']
        
        # ========================================
        # 防御性修复：双重保险
        # ========================================
        if pred_depth.dim() == 4 and pred_depth.shape[1] != 1:
            print(f"⚠️  Warning: pred_depth shape {pred_depth.shape}, taking first channel")
            pred_depth = pred_depth[:, 0:1, :, :]
        
        # 确保形状匹配
        if pred_depth.shape != depth.shape:
            pred_depth = pred_depth.view(depth.shape)
        
        # 填充缺失区域
        completed_depth = depth * (1 - mask_missing) + pred_depth * mask_missing
        
        # 分割模型需要 [B,H,W] 输入
        depth_for_seg = completed_depth.squeeze(1)
        
        # 后续分割操作
        segmentation = self.segmentation_model(rgb, depth_for_seg)
        return pred_depth, completed_depth, segmentation

# ==================== 多任务损失 ====================
# class MultiTaskLoss(nn.Module):
#     def __init__(self, seg_weight=1.0, depth_weight=1.0, class_weights=None,
#                  grad_weight=0.05, mid_weight=0.0):
#         super().__init__()
#         self.seg_weight = seg_weight
#         self.depth_weight = depth_weight
#         self.grad_weight = grad_weight          # ∇ 一致性权重
#         self.mid_weight = mid_weight            # 中频权重（可选）

#         self.seg_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
#         self.l1_loss = nn.L1Loss()

#     # ---------- 梯度一致项 ----------
#     def gradient_loss(self, pred, gt):
#         # Sobel 核
#         sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
#                                dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
#         sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#                                dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

#         def sobel(img):
#             gx = F.conv2d(img, sobel_x, padding=1)
#             gy = F.conv2d(img, sobel_y, padding=1)
#             return torch.abs(gx) + torch.abs(gy)

#         grad_pred = sobel(pred)
#         grad_gt = sobel(gt)
#         return self.l1_loss(grad_pred, grad_gt)

#     # ---------- 中频 loss（LoG 核） ----------
#     def mid_frequency_loss(self, pred, gt):
#         lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
#                            dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
#         mid_pred = F.conv2d(pred, lap, padding=1)
#         mid_gt = F.conv2d(gt, lap, padding=1)
#         return self.l1_loss(mid_pred, mid_gt)
        
#     def forward(self, completed_depth, seg_out, target_depth, seg_label, mask_missing):
#         # 深度损失仅针对缺失区域
#         depth_loss = self.l1_loss(completed_depth * mask_missing, target_depth * mask_missing)
#         seg_loss = self.seg_criterion(seg_out, seg_label)
#         # 统一 4 维
#         if completed_depth.dim() == 3:
#             completed_depth = completed_depth.unsqueeze(1)
#         if target_depth.dim() == 3:
#             target_depth = target_depth.unsqueeze(1)

#         # ② 梯度一致项
#         grad_loss = self.gradient_loss(completed_depth * mask_missing.type(torch.cuda.DoubleTensor),
#                                        target_depth * mask_missing.type(torch.cuda.DoubleTensor))

#         # ③ 中频（可选）
#         mid_loss = self.mid_frequency_loss(completed_depth.type(torch.cuda.DoubleTensor), 
#                                            target_depth.type(torch.cuda.DoubleTensor)) if self.mid_weight > 0 else 0.0

#         # ⑤ 加权总和
#         total = (self.depth_weight * depth_loss +
#                  self.grad_weight * grad_loss +
#                  self.mid_weight * mid_loss +
#                  self.seg_weight * seg_loss)

#         return total, depth_loss, seg_loss
class MultiTaskLoss(nn.Module):
    def __init__(self, seg_weight=1.0, depth_weight=1.0, **kwargs):
        super().__init__()
        self.seg_weight = seg_weight
        self.depth_weight = depth_weight
        self.l1_loss = nn.L1Loss(reduction='none') # 改为 none 手动处理 mask
        self.seg_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, pred_depth, completed_depth, seg_out, target_depth, seg_label, mask_missing):
        # valid_mask: 1.0 表示原始深度图中有值的地方
        valid_mask = 1.0 - mask_missing 
        
        # --- 深度损失计算 ---
        # 我们只在有真值(valid_mask)的地方计算 L1 损失
        # 即使模型输出 0，target_depth 有值，loss 就不再是 0
        diff = self.l1_loss(pred_depth, target_depth) 
        depth_loss = torch.sum(diff * valid_mask) / (torch.sum(valid_mask) + 1e-8)

        # --- 分割损失 ---
        seg_loss = self.seg_criterion(seg_out, seg_label)

        # 总损失
        total = self.depth_weight * depth_loss + self.seg_weight * seg_loss
        
        return total, depth_loss, seg_loss

# ==================== 16-bit 数据集 ====================
class EndToEndDataset(data.Dataset):
    def __init__(self, ids, rgb_folder, depth_folder, label_folder, target_size=(256, 256)):
        self.ids = ids
        self.rgb_fmt = rgb_folder
        self.depth_fmt = depth_folder
        self.label_fmt = label_folder
        self.target_size = target_size
        
        # 预定义最大深度（单位：毫米，假设 16-bit 深度图 1000 代表 1米）
        # 请根据你的数据集说明调整此值，常见的有 10000 (10m) 或 65535 (最大量程)
        self.MAX_DEPTH_VAL = 65535

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        # 1. RGB 处理 (维持原样)
        rgb = Image.open(self.rgb_fmt.format(id_)).convert('RGB')
        rgb = rgb.resize(self.target_size, Image.BICUBIC)
        rgb_tensor = self.rgb_transform(rgb)

        # 2. 16-bit 深度图处理 (重点改进)
        depth16 = cv2.imread(self.depth_fmt.format(id_), cv2.IMREAD_ANYDEPTH)
        if depth16 is None:
            raise FileNotFoundError(f"无法读取深度图: {id_}")
        
        # 使用 INTER_NEAREST 防止在缺失边缘产生插值噪点
        depth16 = cv2.resize(depth16, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # 创建缺失区域掩码 (0 表示缺失)
        mask_missing = (depth16 == 0).astype(np.float32) 
        
        # 物理距离归一化：将毫米/原始数值映射到 [0, 1]
        # 这种方式保证了不同图片的相同深度具有相同的 Tensor 值
        depth_arr = depth16.astype(np.float32) / self.MAX_DEPTH_VAL
        
        # 剪裁超过量程的值，确保在 [0, 1] 范围内
        depth_arr = np.clip(depth_arr, 0.0, 1.0)

        depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0)  # [1, H, W]
        mask_missing_tensor = torch.from_numpy(mask_missing).unsqueeze(0) # [1, H, W]

        # 3. 语义标签 (维持原样)
        label = Image.open(self.label_fmt.format(id_)).convert('L')
        label = label.resize(self.target_size, Image.NEAREST)
        label_tensor = torch.from_numpy(np.array(label, dtype=np.uint8)).long()

        return rgb_tensor, depth_tensor, label_tensor, mask_missing_tensor
    def __init__(self, ids, rgb_folder, depth_folder, label_folder,
                 transform=None, target_size=(256, 256)):
        self.ids = ids
        self.rgb_fmt = rgb_folder
        self.depth_fmt = depth_folder
        self.label_fmt = label_folder
        self.target_size = target_size
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        # 1. RGB 用 PIL
        rgb = Image.open(self.rgb_fmt.format(id_)).convert('RGB')
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=1))
        rgb = rgb.resize(self.target_size, Image.BICUBIC)

        # 2. 深度图用 cv2 读 16-bit
        depth16 = cv2.imread(self.depth_fmt.format(id_), cv2.IMREAD_ANYDEPTH)  # uint16
        if depth16 is None:
            raise FileNotFoundError(self.depth_fmt.format(id_))
        depth16 = cv2.resize(depth16, self.target_size, interpolation=cv2.INTER_NEAREST)

        # 归一化到 0-1（用全局最大最小，可改成数据集统计值）
        d_min, d_max = depth16.min(), depth16.max()
        depth_arr = (depth16.astype(np.float32) - d_min) / (d_max - d_min + 1e-8)

        # 3. 语义标签用 PIL
        label = Image.open(self.label_fmt.format(id_)).convert('L')
        label = label.resize(self.target_size, Image.NEAREST)
        label_arr = np.array(label, dtype=np.uint8)
        
        # 4. 转 tensor
        rgb_tensor = self.transform(rgb)                       # [3,H,W]
        depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0)  # [1,H,W]
        label_tensor = torch.from_numpy(label_arr).long()        # [H,W]
        # print('rgb dtype:', rgb_tensor.dtype)   # 应该是 torch.float32
        # print('depth dtype:', depth_tensor.dtype)   # 应该是 torch.float32
        mask_missing = (depth16 == 0).astype(np.float32)  # [H, W], 缺失处为1，否则为0
        mask_valid = 1 - mask_missing  # 有效区域掩码

        # 将mask转换为tensor
        mask_missing_tensor = torch.from_numpy(mask_missing).unsqueeze(0)  # [1, H, W]
        return rgb_tensor, depth_tensor, label_tensor, mask_missing_tensor

def read_txt_to_list(file_path):
    """读取文本文件到列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [line.strip() for line in lines if line.strip()]
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return []

# ==================== 工具函数 ====================
def read_txt(file):
    with open(file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# ==================== 必须导入的项目模块 ====================
# 请确保这些文件在你的 python path 下，或者和脚本在同一目录
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# ==================== 配置参数 ====================
# ==================== 修复后的测试配置与函数 ====================

class TestConfig:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 8         
        self.num_classes = 16       
        self.image_size = 256       
        
        # 路径配置
        self.data_root = "/root/autodl-tmp/dataset"
        self.test_list_path = "test.txt"
        self.checkpoint_path = "./checkpoints_end2end_new/best_model.pth" # 确保路径正确
        self.save_dir = "./test_results_prediction_BICUBIC"
        
        self.save_prediction_images = True
        self.save_colored_mask = True
        
        # RDFCGAN 的配置文件路径 (必须存在)
        self.gan_config_path = "rdf_cycle_patchgan_config.yaml"

        # 类别定义
        self.class_names = [
            "背景", "类别1", "类别2", "类别3", "类别4", "类别5", 
            "类别6", "类别7", "类别8", "类别9", "类别10", 
            "类别11", "类别12", "类别13", "类别14", "类别15"
        ]
        
        self.class_colors = np.array([
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
            [64, 0, 0], [128, 64, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128]
        ], dtype=np.uint8)

def test_model():
    cfg_test = TestConfig()
    
    # 1. 设置环境
    if not os.path.exists(cfg_test.save_dir) and cfg_test.save_prediction_images:
        os.makedirs(cfg_test.save_dir, exist_ok=True)
        print(f"📁 创建保存目录: {cfg_test.save_dir}")

    # 2. 读取测试列表
    test_ids = read_txt_to_list(cfg_test.test_list_path)
    print(f"🔍 测试集包含 {len(test_ids)} 张图像")

    # 3. 构建模型 (必须与 main() 中的构建逻辑一致！)
    print("🏗️  正在构建模型 (RDFCGAN + ViT_seg)...")
    
    # --- A. 构建深度补全模型 (RDFCGAN) ---
    # 加载 GAN 配置
    load_config(cfg, cfg_test.gan_config_path)
    # 注意：这里不需要再调用 cfg.merge_from_list，因为测试时不修改 save_dir 也没关系
    depth_model = build_model(cfg, cfg_test.device)
    
    # --- B. 构建语义分割模型 (ViT_seg) ---
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = cfg_test.num_classes
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(cfg_test.image_size / 16), int(cfg_test.image_size / 16))
    seg_model = ViT_seg(config_vit, img_size=cfg_test.image_size, num_classes=cfg_test.num_classes).to(cfg_test.device)
    
    # --- C. 组合端到端模型 ---
    # 关键修改：使用你在脚本开头定义的 EndToEndDepthCompletionSegmentation 类
    # 不要重新定义 EndToEndModel 类，因为变量名不一样会导致加载权重失败
    model = EndToEndDepthCompletionSegmentation(
        depth_completion_model=depth_model,
        segmentation_model=seg_model
    ).to(cfg_test.device)

    # 4. 加载权重
    if os.path.exists(cfg_test.checkpoint_path):
        print(f"📥 加载权重: {cfg_test.checkpoint_path}")
        try:
            # 添加 weights_only=False 解决 PyTorch 2.6+ 兼容性问题
            checkpoint = torch.load(cfg_test.checkpoint_path, map_location=cfg_test.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ 权重加载成功！")
        except Exception as e:
            print(f"❌ 加载权重失败: {str(e)}")
            return
    else:
        print(f"❌ 错误: 找不到权重文件 {cfg_test.checkpoint_path}")
        return

    model.eval()

    # 5. 数据加载器
    transform = transforms.Compose([
        transforms.Resize((cfg_test.image_size, cfg_test.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 使用你定义的 TestDataset (确保该类在之前已定义)
    test_dataset = TestDataset(test_ids, cfg_test.data_root, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg_test.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 6. 开始推理
    print("🚀 开始测试...")
    total_conf_matrix = np.zeros((cfg_test.num_classes, cfg_test.num_classes))
    
    with torch.no_grad():
        for batch_ids, images, depths, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(cfg_test.device)
            depths = depths.to(cfg_test.device)
            labels_np = labels.numpy()

            # 前向传播 (depths 可能需要维度调整，EndToEndDepthCompletionSegmentation 内部已处理)
            completed_depth, seg_logits = model(images, depths)
            
            # 获取预测结果
            preds = torch.argmax(seg_logits, dim=1).cpu().numpy()

            # 更新混淆矩阵
            for i in range(len(batch_ids)):
                pred_flat = preds[i].flatten()
                label_flat = labels_np[i].flatten()
                
                valid_mask = label_flat != 255
                cm = confusion_matrix(
                    label_flat[valid_mask], 
                    pred_flat[valid_mask], 
                    labels=range(cfg_test.num_classes)
                )
                total_conf_matrix += cm
                # 保存结果图片 —— 可视化三联动子图
                if cfg_test.save_prediction_images:
                    file_id = batch_ids[i]

                    # 1. 复原 RGB 原图（反归一化到 0-255）
                    img_np = (images[i].cpu().numpy().transpose(1, 2, 0) *
                              np.array([0.229, 0.224, 0.225]) +
                              np.array([0.485, 0.456, 0.406]))
                    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

                    # 2. GT 彩色掩码
                    gt_color = colorize_mask(labels_np[i].astype(np.uint8), cfg_test.class_colors)

                    # 3. Pred 彩色掩码
                    pred_color = colorize_mask(preds[i].astype(np.uint8), cfg_test.class_colors)

                    # 4. 绘制三子图
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(img_np)
                    plt.axis('off')
                    plt.title('RGB Image')

                    plt.subplot(1, 3, 2)
                    plt.imshow(gt_color)
                    plt.axis('off')
                    plt.title('Ground Truth')

                    plt.subplot(1, 3, 3)
                    plt.imshow(pred_color)
                    plt.axis('off')
                    plt.title('Prediction')

                    plt.tight_layout()
                    overview_path = os.path.join(cfg_test.save_dir, f'{file_id}_color_pred.png')
                    plt.savefig(overview_path, bbox_inches='tight', dpi=150)
                    plt.close()          # 防止内存泄漏

                    # （可选）如果仍想要单独保存预测掩码，保留下面两行
                    # pred_color.save(os.path.join(cfg_test.save_dir, f'{file_id}_color_pred.png'))

    # 7. 计算指标
    pixel_acc, mIoU, class_iou = calculate_metrics(total_conf_matrix)

    print("\n" + "="*20 + " 测试结果 " + "="*20)
    print(f"📊 总体准确率 (Pixel Accuracy): {pixel_acc:.4f}")
    print(f"🎯 平均交并比 (mIoU)          : {mIoU:.4f}")
    print("-" * 50)
    
    results_list = []
    for i, iou in enumerate(class_iou):
        class_name = cfg_test.class_names[i] if i < len(cfg_test.class_names) else str(i)
        print(f"{i:<10} {class_name:<25} {iou:.4f}")
        results_list.append({"class_id": i, "name": class_name, "iou": float(iou)})

    # 8. 保存指标到 JSON
    metrics_data = {
        "mIoU": float(mIoU),
        "Pixel_Accuracy": float(pixel_acc),
        "Per_Class_IoU": results_list
    }
    with open(os.path.join(cfg_test.save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 测试完成！结果已保存至 {cfg_test.save_dir}")
# ==================== 数据集 ====================
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, ids, root_dir, transform=None):
        self.ids = ids
        self.root_dir = root_dir
        self.transform = transform
        
        # 预定义子文件夹格式
        self.image_path = os.path.join(root_dir, "Image", "{}.jpg")
        self.depth_path = os.path.join(root_dir, "Depth", "{}.png")
        self.label_path = os.path.join(root_dir, "Label_Label", "{}_label.png")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file_id = self.ids[idx]
        
        # 1. 加载图像
        img = Image.open(self.image_path.format(file_id)).convert('RGB')
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        # 2. 加载深度图 (转为 Tensor)
        dsm = cv2.imread(self.depth_path.format(file_id), cv2.IMREAD_ANYDEPTH)  # uint16
        if dsm is None:
            raise FileNotFoundError(self.depth_path.format(file_id))
        dsm = cv2.resize(dsm, (256, 256), interpolation=cv2.INTER_NEAREST)

        # 归一化到 0-1（用全局最大最小，可改成数据集统计值）
        d_min, d_max = dsm.min(), dsm.max()
        dsm_arr = (dsm.astype(np.float32) - d_min) / (d_max - d_min + 1e-8)

        dsm_tensor = torch.from_numpy(dsm_arr).unsqueeze(0) # [1, H, W]
        
        # 3. 加载标签 (用于计算指标)
        label = Image.open(self.label_path.format(file_id)).convert('L')
        label = label.resize((256, 256), Image.NEAREST) # 标签必须用最近邻插值
        label_tensor = torch.from_numpy(np.array(label, dtype=np.int64))
        
        # 4. 图像变换
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
            
        return file_id, img_tensor, dsm_tensor, label_tensor

# ==================== 辅助函数 ====================
def colorize_mask(mask, palette):
    """将索引掩码转换为彩色图像"""
    # mask: [H, W] numpy array
    # palette: [num_classes, 3] numpy array
    color_img = palette[mask]
    return Image.fromarray(color_img)

def calculate_metrics(conf_matrix):
    """根据混淆矩阵计算 IoU 和 mIoU"""
    # 避免除以0
    pos = conf_matrix.sum(1)
    res = conf_matrix.sum(0)
    tp = np.diag(conf_matrix)
    
    pixel_acc = tp.sum() / conf_matrix.sum()
    
    # Intersection over Union
    # IoU = TP / (TP + FP + FN)
    iou_array = tp / np.maximum(1.0, pos + res - tp)
    mean_iou = np.nanmean(iou_array)
    
    return pixel_acc, mean_iou, iou_array

if __name__ == "__main__":
    #test_model()
    staged_training()
