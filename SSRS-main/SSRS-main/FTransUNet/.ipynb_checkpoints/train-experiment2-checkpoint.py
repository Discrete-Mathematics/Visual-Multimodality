import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random
import time
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from torch.autograd import Variable
from IPython.display import clear_output
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import urllib.request
import os
import psutil, gc
import torchvision.transforms as transforms
from PIL import Image
import torch.cuda.amp as amp  # 添加混合精度支持
import matplotlib.font_manager as fm
import preprocess as p
import json
import move_label as ml
# 设置字体
#plt.rcParams['font.family'] = 'SimHei'
#plt.rcParams['axes.unicode_minus'] = False

os.makedirs('./experiment3', exist_ok=True)

# 优化PyTorch显存分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 修改数据集路径配置
DATA_FOLDER = "/root/autodl-tmp/dataset/cropped_Image/{}.jpg"
LABEL_FOLDER = "/root/autodl-tmp/dataset/cropped_Label/{}_label.png"
DSM_FOLDER = "/root/autodl-tmp/dataset/cropped_Depth/{}.png"
N_CLASSES = 16
BATCH_SIZE = 32  # 根据GPU内存调整
Stride_Size = 128
WINDOW_SIZE = (256, 256)

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = N_CLASSES
config_vit.n_skip = 3
config_vit.patches.grid = (int(256 / 16), int(256 / 16))

# 创建模型并加载预训练权重
net = ViT_seg(config_vit, img_size=256, num_classes=16).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path, allow_pickle=True))

# 打印模型参数数量
params = sum(p.numel() for p in net.parameters())
print(f"模型参数数量: {params}")

# 优化器配置
base_lr = 0.01
optimizer = optim.Adam(net.parameters(), lr=base_lr, betas=(0.8, 0.999), eps=1e-8, weight_decay=0.0005, amsgrad=True)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150], gamma=0.5)

# 创建混合精度训练的scaler
scaler = amp.GradScaler()

def test(net, test_loader, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    """
    修复后的测试函数 - 使用数据加载器而不是文件ID列表
    """
    net.eval()
    all_preds = []
    all_gts = []

    with torch.no_grad(), amp.autocast():
        for batch_idx, (img, dsm, gt) in enumerate(tqdm(test_loader, desc="测试中", leave=False)):
            # 移动到GPU
            img = img.cuda(non_blocking=True)
            dsm = dsm.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            dsm = dsm.squeeze(1)
            # 前向传播
            outputs = net(img, dsm)
            
            # 获取预测结果
            _, preds = torch.max(outputs, 1)
            
            # 转移到CPU并添加到结果列表
            all_preds.append(preds.cpu().numpy())
            all_gts.append(gt.cpu().numpy())
            
            # 及时释放内存
            del img, dsm, gt, outputs, preds
            torch.cuda.empty_cache()
    
    # 展平所有预测和真实标签
    all_preds_flat = np.concatenate([p.ravel() for p in all_preds])
    all_gts_flat = np.concatenate([p.ravel() for p in all_gts])
    
    # 计算精度
    pa, cpa, mpa, mIoU = metrics(all_preds_flat, all_gts_flat)
    
    if all:
        return pa, cpa, mpa, mIoU, all_preds, all_gts
    else:
        return pa, cpa, mpa, mIoU

def train(net, optimizer, train_loader, test_loader, epochs, scheduler=None, weights=None, save_epoch=1):
    """
    修复后的训练函数 - 传入数据加载器参数
    """
    # 初始化损失记录
    losses = []
    best_acc = 0.0
    
    # 权重处理
    if weights is not None:
        weights = weights.cuda()
    else:
        weights = torch.ones(config_vit.n_classes).cuda()
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    for epoch in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        
        net.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            # 异步传输到GPU
            data = data.cuda(non_blocking=True)
            data = data.float()
            
            dsm = dsm.cuda(non_blocking=True)
            dsm = dsm.squeeze(1)
            dsm = dsm.float()
            
            target = target.cuda(non_blocking=True)
            target = target.squeeze(1)
            target = target.long()
            
            optimizer.zero_grad()
            
            # 混合精度训练
            with amp.autocast():
                output = net(data, dsm)
                loss = criterion(output, target)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 记录损失
            epoch_loss += loss.item()
            
            # 打印训练进度
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    acc = (preds == target).float().mean().item()
                print(f'Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} '
                      f'| Loss: {loss.item():.4f} | Acc: {acc:.4f}')
            
            # 及时释放内存
            del data, dsm, target, output
            torch.cuda.empty_cache()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f}')
        
        # 定期测试并保存模型
        if epoch % save_epoch == 0 or epoch == epochs:
            net.eval()
            pa, cpa, mpa, mIoU = test(net, test_loader, all=False, stride=Stride_Size)  # 修复：使用test_loader
            net.train()
            
            print(f'测试结果 - PA: {pa:.4f}\nCPA: {cpa}\nmPA: {mpa:.4f}\nmIoU: {mIoU:.4f}')
            
            if pa > best_acc:
                best_acc = pa  # 修复：使用pa而不是acc
                torch.save(net.state_dict(), f'./experiment3/best_model_epoch{epoch}_acc{pa:.4f}.pth')
                print(f'保存最佳模型，准确率: {pa:.4f}')
    
    print(f'最佳准确率: {best_acc:.4f}')
    return best_acc

# 优化的数据集类（内存友好）
class MemoryFriendlyDataset(torch.utils.data.Dataset):
    def __init__(self, ids, cache=False):
        """
        内存友好的数据集类
        - 不缓存整个数据集，只存储文件路径
        - 按需加载单个样本
        """
        self.ids = ids
        self.cache = cache
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        # 内存低于500MB时强制清理
        if psutil.virtual_memory().available < 500 * 1024**2:
            gc.collect()
            torch.cuda.empty_cache()
            
        id = self.ids[index]
        
        # 加载图像
        img = Image.open(DATA_FOLDER.format(id)).convert('RGB')
        img_tensor = self.transform(img)
        
        # 加载标签
        lbl = Image.open(LABEL_FOLDER.format(id))
        lbl_tensor = torch.tensor(np.array(lbl), dtype=torch.long)
        
        # 加载并处理DSM
        dsm = Image.open(DSM_FOLDER.format(id))
        dsm_tensor = torch.tensor(np.array(dsm), dtype=torch.float32)
        
        # 归一化DSM
        min_val = torch.min(dsm_tensor)
        max_val = torch.max(dsm_tensor)
        if max_val - min_val < 1e-6:
            dsm_tensor = torch.zeros_like(dsm_tensor)
        else:
            dsm_tensor = (dsm_tensor - min_val) / (max_val - min_val)
        
        # 添加通道维度
        dsm_tensor = dsm_tensor.unsqueeze(0)
        return img_tensor, dsm_tensor, lbl_tensor

# 主程序
if __name__ == "__main__":
    # 准备数据
    train_set, test_set = p.prepare_multimodal_datasets()
    
    print(f"训练样本数: {len(train_set)}")
    print(f"验证样本数: {len(test_set)}")
    
    # 可视化训练增强效果
    print("\n=== 多模态训练数据增强示例 ===")
    p.display_multimodal_augmentations(train_set, p.cfg.VISUALIZE_SAMPLES)
        
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,           # 多进程加载
        pin_memory=True,          # 使用锁页内存加速传输
        persistent_workers=True   # 保持工作进程
    )
    
    # 创建测试数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 测试一个批次
    batch_m1, batch_m2, batch_masks = next(iter(train_loader))
    print(f"批次模态1尺寸: {batch_m1.shape}")
    print(f"批次模态2尺寸: {batch_m2.shape}")
    print(f"批次Mask尺寸: {batch_masks.shape}")
    print("多模态数据管道测试完成！")
    
    # 开始训练 - 修复：传入数据加载器
    start_time = time.time()
    best_pa = train(net, optimizer, train_loader, test_loader, epochs=205, scheduler=scheduler)
    end_time = time.time()
    print(f'总耗时: {(end_time - start_time)/60:.2f} 分钟')
    
    # 加载最佳模型进行最终测试
    try:
        best_model_path = max(glob('./experiment3/best_model_*.pth'), key=os.path.getctime)
        net.load_state_dict(torch.load(best_model_path))
        print(f"加载最佳模型: {best_model_path}")
    except ValueError:
        print("未找到保存的模型文件，使用当前模型进行测试")
    
    # 最终测试
    net.eval()
    pa, cpa, mpa, mIoU, all_preds, all_gts = test(net, test_loader, all=True, stride=32)
    print(f"最终测试pa: {pa:.4f}")
    print(f"最终测试cpa: {cpa}")
    print(f"最终测试mpa: {mpa:.4f}")
    print(f"最终测试mIoU: {mIoU:.4f}")

    # 保存结果信息
    info = {
        "批次模态1尺寸": str(batch_m1.shape),  # 转为字符串以便JSON序列化
        "批次模态2尺寸": str(batch_m2.shape), 
        "批次Mask尺寸": str(batch_masks.shape), 
        "best_pa": float(best_pa),
        "最终测试pa": float(pa),
        "最终测试cpa": float(cpa),
        "最终测试mpa": float(mpa),
        "最终测试mIoU": float(mIoU)
    }
    
    with open("info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    # 可视化结果 - 修复可视化部分
    print("开始可视化预测结果...")
    for i, (batch_imgs, batch_dsms, batch_gts) in enumerate(test_loader):
        if i >= 5:  # 只可视化前5个批次的第一个样本
            break
            
        # 取第一个样本进行可视化
        img_tensor = batch_imgs[0]
        gt_tensor = batch_gts[0]
        
        # 获取对应的预测结果
        if i < len(all_preds):
            pred = all_preds[i][0] if len(all_preds[i]) > 0 else all_preds[i]
        else:
            continue
            
        plt.figure(figsize=(12, 4))
        
        # 原始图像 - 转换tensor为numpy并调整维度顺序
        plt.subplot(1, 3, 1)
        img_np = img_tensor.permute(1, 2, 0).numpy()  # CHW -> HWC
        plt.imshow(img_np)
        plt.title('Original Image')
        plt.axis('off')
        
        # 真实标签
        plt.subplot(1, 3, 2)
        gt_np = gt_tensor.numpy()
        plt.imshow(gt_np, cmap='tab20')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # 预测结果
        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap='tab20')
        plt.title('Prediction')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'./experiment3/result_batch_{i}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    print("可视化完成！")