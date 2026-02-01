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
from PIL import Image
import matplotlib.pyplot as plt

# ==================== 模型导入 ====================
from gan import build_model
from lib.utils.configurator import cfg, load_config
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg

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

    def forward(self, rgb, depth):
        depth = depth.to(dtype=rgb.dtype, device=rgb.device)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)                  # [B,1,H,W]
        raw_depth = depth.squeeze(1)                    # [B,H,W]

        ret = self.depth_completion_model.forward_test(rgb=rgb, raw_depth=raw_depth)
        pred_depth = ret['pred_depth'].squeeze(1)       # [B,H,W]
        completed_depth = torch.clamp(pred_depth.unsqueeze(1), 0, 1)  # [B,1,H,W]

        segmentation = self.segmentation_model(rgb, completed_depth.squeeze(1))
        return completed_depth, segmentation


# ==================== 多任务损失 ====================
class MultiTaskLoss(nn.Module):
    def __init__(self, seg_weight=1.0, depth_weight=1.0, class_weights=None):
        super().__init__()
        self.seg_weight = seg_weight
        self.depth_weight = depth_weight
        self.seg_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        self.l1_loss = nn.SmoothL1Loss(beta=0.1)

    def forward(self, completed_depth, seg_out, target_depth, seg_label):
        # 统一 4 维
        if completed_depth.dim() == 3:
            completed_depth = completed_depth.unsqueeze(1)
        if target_depth.dim() == 3:
            target_depth = target_depth.unsqueeze(1)

        depth_loss = self.l1_loss(completed_depth, target_depth)
        seg_loss = self.seg_criterion(seg_out, seg_label)
        total = self.depth_weight * depth_loss + self.seg_weight * seg_loss
        return total, depth_loss, seg_loss


# ==================== 16-bit 数据集 ====================
class EndToEndDataset(data.Dataset):
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
        rgb = rgb.resize(self.target_size, Image.BILINEAR)

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
        return rgb_tensor, depth_tensor, label_tensor


# ==================== Trainer（与之前一致，仅拷贝骨架） ====================
class EndToEndTrainer:
    def __init__(self, model, train_loader, val_loader, device,
                 save_dir='./checkpoints_end2end', resume_path=None):
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
        self.criterion = MultiTaskLoss(seg_weight=1.0, depth_weight=0.5)
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
        for rgb, depth, labels in pbar:
            rgb, depth, labels = rgb.to(self.device), depth.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with amp.autocast():
                pred_depth, seg = self.model(rgb, depth)
                loss, depth_loss, seg_loss = self.criterion(pred_depth, seg, depth, labels)
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
            for rgb, depth, labels in tqdm(self.val_loader, desc='验证'):
                rgb, depth, labels = rgb.to(self.device), depth.to(self.device), labels.to(self.device)
                pred_depth, seg = self.model(rgb, depth)
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


# ==================== 工具函数 ====================
def read_txt(file):
    with open(file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


# ==================== main ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    DATA_ROOT = '/root/autodl-tmp/dataset'
    rgb_fmt   = os.path.join(DATA_ROOT, 'Image', '{}.jpg')
    depth_fmt = os.path.join(DATA_ROOT, 'Depth', '{}.png')
    label_fmt = os.path.join(DATA_ROOT, 'Label_Label', '{}_label.png')

    train_ids = read_txt('train.txt')
    val_ids   = read_txt('val.txt')

    train_set = EndToEndDataset(train_ids, rgb_fmt, depth_fmt, label_fmt,
                                target_size=(256, 256))
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

    # 训练器 & 启动
    trainer = EndToEndTrainer(model, train_loader, val_loader, device,
                              save_dir='./checkpoints_end2end_new',
                              resume_path=None)   # 或给 checkpoint 路径
    trainer.train(epochs=500)

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
        self.checkpoint_path = "./best_model.pth" # 确保路径正确
        self.save_dir = "./test_results_prediction"
        
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
def read_txt_to_list(file_path):
    """读取文本文件到列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [line.strip() for line in lines if line.strip()]
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return []
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
    test_model()
    #main()
