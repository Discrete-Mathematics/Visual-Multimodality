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
# 设置字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('./experime', exist_ok=True)

# 优化PyTorch显存分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 修改数据集路径配置
DATA_FOLDER = "/root/autodl-tmp/dataset/cropped_Image/{}.jpg"
LABEL_FOLDER = "/root/autodl-tmp/dataset/cropped_Label/{}_label.png"
DSM_FOLDER = "/root/autodl-tmp/dataset/cropped_Depth/{}.png"
N_CLASSES = 16
BATCH_SIZE = 12  # 根据GPU内存调整
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

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    """测试函数 - 使用混合精度推理"""
    net.eval()
    all_preds = []
    all_gts = []

    with torch.no_grad(), amp.autocast():
        for id in tqdm(test_ids, desc="测试中", leave=False):
            # 加载测试数据
            img = np.asarray(Image.open(DATA_FOLDER.format(id)).convert('RGB'), dtype=np.float32) / 255.0
            dsm = np.asarray(Image.open(DSM_FOLDER.format(id)), dtype=np.float32)
            gt = np.asarray(Image.open(LABEL_FOLDER.format(id)), dtype=np.uint8)
            
            # 归一化DSM
            min_val = np.min(dsm)
            max_val = np.max(dsm)
            if max_val - min_val < 1e-6:
                dsm = np.zeros_like(dsm)
            else:
                dsm = (dsm - min_val) / (max_val - min_val)
            
            pred = np.zeros(img.shape[:2] + (config_vit.n_classes,))
            
            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), 
                                          total=total, leave=False)):
                # 准备批次数据
                image_patches = []
                dsm_patches = []
                
                for x, y, w, h in coords:
                    img_patch = img[x:x+w, y:y+h].transpose((2, 0, 1))
                    dsm_patch = dsm[x:x+w, y:y+h]
                    image_patches.append(img_patch)
                    dsm_patches.append(dsm_patch)
                
                # 转换为张量并移动到GPU
                image_tensor = torch.tensor(np.array(image_patches), dtype=torch.float32).cuda(non_blocking=True)
                dsm_tensor = torch.tensor(np.array(dsm_patches), dtype=torch.float32).cuda(non_blocking=True)
                
                # 推理
                outs = net(image_tensor, dsm_tensor)
                outs = outs.cpu().numpy()
                
                # 填充预测结果
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x+w, y:y+h] += out
                
                # 及时释放内存
                del image_tensor, dsm_tensor, outs
                torch.cuda.empty_cache()
            
            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt)
    
    # 计算精度
    pa, cpa, mpa, mIoU = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return pa, cpa, mpa, mIoU, all_preds, all_gts
    else:
        return pa, cpa, mpa, mIoU

def train(net, optimizer, epochs, scheduler=None, weights=None, save_epoch=1):
    """训练函数 - 使用混合精度训练"""
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
            dsm = dsm.cuda(non_blocking=True)
            dsm = dsm.squeeze(1)
            target = target.cuda(non_blocking=True)
            
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
            pa, cpa, mpa, mIoU = test(net, test_ids, all=False, stride=Stride_Size)
            net.train()
            
            if pa > best_acc:
                best_acc = acc
                torch.save(net.state_dict(), f'./experiment2/best_model_epoch{epoch}_acc{pa:.2f}.pth')
                print(f'保存最佳模型，准确率: {pa:.2f}')
    
    print(f'最佳准确率: {best_acc:.2f}')

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
    # 获取文件列表
    TARGET_DIR = "/root/autodl-tmp/dataset/cropped_Depth"
    all_ids = [os.path.splitext(f)[0] for f in os.listdir(TARGET_DIR) if f.endswith('.png')]
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(all_ids))
    test_size = len(all_ids) - train_size
    train_ids, test_ids = torch.utils.data.random_split(all_ids, [train_size, test_size])
    
    print(f"训练集大小: {len(train_ids)}")
    print(f"测试集大小: {len(test_ids)}")
    
    # 数据集参数
    BATCH_SIZE = 10  # 根据GPU内存调整
    Stride_Size = 128
    WINDOW_SIZE = (256, 256)
    
    # 创建数据集和数据加载器
    train_set = MemoryFriendlyDataset(train_ids.dataset)
    test_set = MemoryFriendlyDataset(test_ids.dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,           # 多进程加载
        pin_memory=True,          # 使用锁页内存加速传输
        persistent_workers=True   # 保持工作进程
    )
    
    # 开始训练
    start_time = time.time()
    train(net, optimizer, epochs=50, scheduler=scheduler)
    end_time = time.time()
    print(f'总耗时: {(end_time - start_time)/60:.2f} 分钟')
    
    # 加载最佳模型进行最终测试
    best_model_path = max(glob('./experiment2/best_model_*.pth'), key=os.path.getctime)
    net.load_state_dict(torch.load(best_model_path))
    
    # 最终测试
    net.eval()
    pa, cpa, mpa, mIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
    print(f"最终测试pa: {pa:.2f}")
    print(f"最终测试cpa: {cpa}")
    print(f"最终测试mpa: {mpa:.2f}")
    print(f"最终测试mIoU: {mIoU:.2f}")
    
    # 可视化结果
    for i, (pred, id) in enumerate(zip(all_preds, test_ids)):
        if i < 5:  # 只可视化前5个结果
            plt.figure(figsize=(12, 4))
            
            # 原始图像
            plt.subplot(1, 3, 1)
            img = np.array(Image.open(DATA_FOLDER.format(id)))
            plt.imshow(img)
            plt.title('original')
            
            # 真实标签
            plt.subplot(1, 3, 2)
            gt = np.array(Image.open(LABEL_FOLDER.format(id)))
            plt.imshow(gt)
            plt.title('ground_truth')
            
            # 预测结果
            plt.subplot(1, 3, 3)
            plt.imshow(pred)
            plt.title('prediction')
            
            plt.tight_layout()
            plt.savefig(f'./experiment2/result_{id}.png')
            plt.show()