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
import torch.cuda.amp as amp
import matplotlib.font_manager as fm
from torchvision.transforms.functional import resize
from scipy.ndimage import gaussian_filter

def read_txt_to_list(file_path, strip_whitespace=True, skip_empty_lines=True):
    """
    读取文本文件并将其内容转换为 Python 列表
    
    参数:
        file_path (str): 文本文件的路径
        strip_whitespace (bool): 是否去除每行首尾的空白字符（默认True）
        skip_empty_lines (bool): 是否跳过空行（默认True）
    
    返回:
        list: 包含文件每行内容的列表
    """
    try:
        # 打开文件并读取内容
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # 处理每行内容
        result = []
        for line in lines:
            # 去除首尾空白字符（如果启用）
            if strip_whitespace:
                line = line.strip()
            
            # 跳过空行（如果启用）
            if skip_empty_lines and not line:
                continue
                
            result.append(line)
        
        return result
    
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
        return []
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return []

train_ids = read_txt_to_list("train.txt")
test_ids = read_txt_to_list("test.txt")
print(f"训练集大小: {len(train_ids)}")
print(f"测试集大小: {len(test_ids)}")
os.makedirs('./experiment6', exist_ok=True)

# 优化PyTorch显存分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 环境检查
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"cuDNN版本: {torch.backends.cudnn.version()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 修改数据集路径配置
DATA_FOLDER = "/root/autodl-tmp/dataset/cropped_Image/{}.jpg"
LABEL_FOLDER = "/root/autodl-tmp/dataset/cropped_Label/{}_label.png"
DSM_FOLDER = "/root/autodl-tmp/dataset/cropped_Depth/{}.png"
N_CLASSES = 16
BATCH_SIZE = 32
EPOCHS = 205
OVERLAP_RATIO = 0.75  # 75%重叠
WINDOW_SIZE = (256, 256)
STRIDE_SIZE = int(WINDOW_SIZE[0] * (1 - OVERLAP_RATIO))  # 128
CONTEXT_MARGIN = 64  # 上下文扩展区域

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = N_CLASSES
config_vit.n_skip = 3
config_vit.patches.grid = (int(256 / 16), int(256 / 16))

# 创建模型并加载预训练权重
net = ViT_seg(config_vit, img_size=256, num_classes=16).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path, allow_pickle=True))

# 检查模型参数
def check_model_parameters(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"警告! 参数 {name} 包含NaN或Inf值")
            # 尝试修复
            param.data = torch.nan_to_num(param.data)
            print(f"已修复参数 {name}")

check_model_parameters(net)

# 打印模型参数数量
params = sum(p.numel() for p in net.parameters())
print(f"模型参数数量: {params}")

# 优化器配置 - 使用AdamW + AMSGrad
base_lr = 0.000005
optimizer = optim.AdamW(
    net.parameters(), 
    lr=base_lr, 
    weight_decay=0.01, 
    betas=(0.9, 0.999),
    eps=1e-6,  # 增加epsilon值提高稳定性
    amsgrad=False
)

# 使用更保守的余弦退火学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=50, 
    eta_min=base_lr/1000  # 更小的最小学习率
)

# 创建混合精度训练的scaler（更保守的设置）
scaler = amp.GradScaler(
    init_scale=4096.0,  # 初始缩放因子
    growth_factor=1.1,    # 更保守的增长因子
    backoff_factor=0.5,   # 更积极的回退因子
    growth_interval=1000  # 更长的增长间隔
)

# 添加梯度裁剪（重要安全措施）
torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

# 安全的交叉熵损失函数
class SafeCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
        
    def forward(self, input, target):
        # 添加输入保护
        input = torch.clamp(input, min=-100, max=100)  # 防止极端值
        
        # 计算标准交叉熵
        loss = F.cross_entropy(input, target, weight=self.weight)
        
        # 如果损失为NaN，返回替代值
        if torch.isnan(loss) or torch.isinf(loss):
            print("交叉熵损失为NaN或Inf，使用替代值")
            return torch.tensor(1.0, device=input.device, requires_grad=True)
            
        return loss

# 改进的测试函数 - 使用高斯加权融合和上下文扩展
def test(net, test_ids, all=False, stride=STRIDE_SIZE, batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    """改进的测试函数 - 使用高斯加权融合和上下文扩展"""
    net.eval()
    all_preds = []
    all_gts = []
    
    # 创建高斯权重矩阵用于融合
    gaussian_weights = np.zeros(window_size)
    center_x, center_y = window_size[0] // 2, window_size[1] // 2
    for i in range(window_size[0]):
        for j in range(window_size[1]):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            gaussian_weights[i, j] = np.exp(-distance**2 / (2 * (window_size[0]/4)**2))
    
    # DSM归一化函数
    def normalize_dsm(dsm_arr):
        epsilon = 1e-8
        min_val = np.min(dsm_arr)
        max_val = np.max(dsm_arr)
        
        # 处理特殊情况
        if max_val - min_val < epsilon:
            return np.zeros_like(dsm_arr)
        
        # 添加数据裁剪防止极端值
        clipped_dsm = np.clip(dsm_arr, min_val, min_val + 10*(max_val-min_val))
        
        return (clipped_dsm - min_val) / (max_val - min_val + epsilon)
    
    with torch.no_grad(), amp.autocast():
        for id in tqdm(test_ids, desc="测试中", leave=False):
            # 加载测试数据
            img = np.asarray(Image.open(DATA_FOLDER.format(id)).convert('RGB'), dtype=np.float32) / 255.0
            dsm = np.asarray(Image.open(DSM_FOLDER.format(id)), dtype=np.float32)
            gt = np.asarray(Image.open(LABEL_FOLDER.format(id)), dtype=np.uint8)
            
            # 归一化DSM
            dsm = normalize_dsm(dsm)
            
            # 初始化融合图和计数图
            pred_map = np.zeros(img.shape[:2] + (config_vit.n_classes,))
            weight_map = np.zeros(img.shape[:2])
            
            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), 
                                          total=total, leave=False)):
                # 准备批次数据
                image_patches = []
                dsm_patches = []
                
                for (x, y, w, h) in coords:
                    # 添加上下文区域
                    x_start = max(0, x - CONTEXT_MARGIN)
                    y_start = max(0, y - CONTEXT_MARGIN)
                    x_end = min(img.shape[0], x + w + CONTEXT_MARGIN)
                    y_end = min(img.shape[1], y + h + CONTEXT_MARGIN)
                    
                    # 提取带上下文的图像块
                    img_patch = img[x_start:x_end, y_start:y_end]
                    dsm_patch = dsm[x_start:x_end, y_start:y_end]
                    
                    # 转换为PIL图像进行大小调整
                    img_patch_pil = Image.fromarray((img_patch * 255).astype(np.uint8))
                    dsm_patch_pil = Image.fromarray((dsm_patch * 255).astype(np.uint8))
                    
                    # 调整到模型输入大小
                    img_patch_resized = img_patch_pil.resize(WINDOW_SIZE, Image.BILINEAR)
                    dsm_patch_resized = dsm_patch_pil.resize(WINDOW_SIZE, Image.BILINEAR)
                    
                    # 转换回数组并归一化
                    img_patch_arr = np.array(img_patch_resized, dtype=np.float32) / 255.0
                    dsm_patch_arr = np.array(dsm_patch_resized, dtype=np.float32) / 255.0
                    
                    image_patches.append(img_patch_arr.transpose((2, 0, 1)))
                    dsm_patches.append(dsm_patch_arr)
                
                # 转换为张量并移动到GPU
                image_tensor = torch.tensor(np.array(image_patches), dtype=torch.float32).cuda(non_blocking=True)
                dsm_tensor = torch.tensor(np.array(dsm_patches), dtype=torch.float32).cuda(non_blocking=True)
                
                # 添加输入检查
                if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
                    print("测试输入包含NaN/Inf值，跳过该批次")
                    continue
                
                # 推理
                outs = net(image_tensor, dsm_tensor)
                outs = outs.cpu().numpy()
                
                # 填充预测结果（使用高斯加权）
                for out, (x, y, w, h) in zip(outs, coords):
                    # 裁剪上下文区域
                    context_x = max(0, x - CONTEXT_MARGIN)
                    context_y = max(0, y - CONTEXT_MARGIN)
                    
                    # 计算裁剪后的位置
                    crop_x = x - context_x
                    crop_y = y - context_y
                    
                    # 裁剪出原始窗口区域
                    out_cropped = out[:, crop_x:crop_x+w, crop_y:crop_y+h].transpose((1, 2, 0))
                    
                    # 应用高斯加权
                    for c in range(config_vit.n_classes):
                        pred_map[x:x+w, y:y+h, c] += out_cropped[:, :, c] * gaussian_weights
                    
                    weight_map[x:x+w, y:y+h] += gaussian_weights
                
                # 及时释放内存
                del image_tensor, dsm_tensor, outs
                torch.cuda.empty_cache()
            
            # 归一化融合结果
            for c in range(config_vit.n_classes):
                pred_map[:, :, c] /= (weight_map + 1e-8)
            
            pred = np.argmax(pred_map, axis=-1)
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
    """改进的训练函数 - 使用多尺度训练和上下文扩展"""
    # 初始化损失记录
    losses = []
    best_mIoU = 0.0
    
    # 权重处理
    if weights is not None:
        weights = weights.cuda()
    else:
        weights = torch.ones(config_vit.n_classes).cuda()
    
    # 使用安全的损失函数
    criterion = SafeCrossEntropyLoss(weight=weights)
    
    for epoch in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
            print(f"当前学习率: {scheduler.get_last_lr()[0]}")
        
        net.train()
        epoch_loss = 0.0
        valid_batches = 0
        
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            # 添加数据检查
            if torch.isnan(data).any() or torch.isnan(dsm).any() or torch.isnan(target).any():
                print(f"批次 {batch_idx} 包含NaN值，跳过")
                continue
                
            # 异步传输到GPU
            data = data.cuda(non_blocking=True)
            dsm = dsm.cuda(non_blocking=True)
            dsm = dsm.squeeze(1)
            target = target.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            
            # 混合精度训练
            with amp.autocast():
                # 添加输入检查
                if torch.isnan(data).any() or torch.isinf(data).any():
                    print(f"批次 {batch_idx} 输入包含NaN/Inf值!")
                    data = torch.nan_to_num(data)
                    
                if torch.isnan(dsm).any() or torch.isinf(dsm).any():
                    print(f"批次 {batch_idx} DSM包含NaN/Inf值!")
                    dsm = torch.nan_to_num(dsm)
                
                output = net(data, dsm)
                
                # 添加输出检查
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"批次 {batch_idx} 模型输出包含NaN/Inf值!")
                    # 尝试修复
                    output = torch.nan_to_num(output)
                    # 如果仍然无效则跳过
                    if torch.isnan(output).any():
                        print("无法修复，跳过批次")
                        optimizer.zero_grad()
                        del data, dsm, target, output
                        torch.cuda.empty_cache()
                        continue
                
                loss = criterion(output, target)
                
                # 添加损失检查
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"批次 {batch_idx} 损失值无效: {loss.item()}")
                    optimizer.zero_grad()
                    del data, dsm, target, output
                    torch.cuda.empty_cache()
                    continue
            
            scaler.scale(loss).backward()
            
            # 更严格的梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0, norm_type=2.0)
            
            # 梯度检查
            grad_nan_detected = False
            for name, param in net.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"参数 {name} 梯度包含NaN值!")
                        param.grad = torch.nan_to_num(param.grad)
                        grad_nan_detected = True
            
            if grad_nan_detected:
                print("检测到NaN梯度，已修复")
            
            scaler.step(optimizer)
            scaler.update()
            
            # 记录详细统计信息
            with torch.no_grad():
                output_mean = output.float().mean().item()
                output_std = output.float().std().item()
                output_max = output.abs().max().item()
            
            epoch_loss += loss.item()
            valid_batches += 1
            
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    acc = (preds == target).float().mean().item()
                print(f'Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} '
                      f'| Loss: {loss.item():.4f} | Acc: {acc:.4f} '
                      f'| Output: mean={output_mean:.4f}, std={output_std:.4f}, max={output_max:.4f}')
            
            del data, dsm, target, output
            torch.cuda.empty_cache()
        
        if valid_batches == 0:
            print(f"Epoch {epoch} 没有有效批次，跳过")
            continue
            
        avg_loss = epoch_loss / valid_batches
        losses.append(avg_loss)
        print(f'Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f} | 有效批次: {valid_batches}/{len(train_loader)}')
        
        # 定期测试并保存模型
        if epoch % save_epoch == 0 or epoch == epochs:
            net.eval()
            pa, cpa, mpa, mIoU = test(net, test_ids, all=False, stride=STRIDE_SIZE)
            net.train()
            print(f"mIoU:{mIoU}")
            if mIoU > best_mIoU:
                best_mIoU = mIoU
                torch.save(net.state_dict(), f'./experiment6/best_model_epoch{epoch}_mIoU{mIoU:.4f}.pth')
                print(f'保存最佳模型，mIoU: {mIoU:.4f}')
    
    print(f'最佳mIoU: {best_mIoU:.4f}')

# DSM归一化函数
def normalize_dsm(dsm_arr):
    epsilon = 1e-8
    min_val = np.min(dsm_arr)
    max_val = np.max(dsm_arr)
    
    # 处理特殊情况
    if max_val - min_val < epsilon:
        return np.zeros_like(dsm_arr)
    
    # 添加数据裁剪防止极端值
    clipped_dsm = np.clip(dsm_arr, min_val, min_val + 10*(max_val-min_val))
    
    return (clipped_dsm - min_val) / (max_val - min_val + epsilon)

# 改进的数据集类（支持多尺度训练和上下文扩展）
class EnhancedDataset(torch.utils.data.Dataset):
    def __init__(self, ids, scales=[0.75, 1.0, 1.25], cache=False):
        """
        增强的数据集类
        - 支持多尺度训练
        - 添加上下文扩展
        - 随机裁剪
        """
        self.ids = ids
        self.cache = cache
        self.scales = scales
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        # 内存低于500MB时强制清理
        if psutil.virtual_memory().available < 500 * 1024**2:
            gc.collect()
            torch.cuda.empty_cache()
            
        id = self.ids[index]
        
        # 随机选择尺度
        scale = random.choice(self.scales)
        
        # 加载图像
        img = Image.open(DATA_FOLDER.format(id)).convert('RGB')
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
        
        # 加载标签
        lbl = Image.open(LABEL_FOLDER.format(id))
        lbl = lbl.resize((int(lbl.width * scale), int(lbl.height * scale)), Image.NEAREST)
        
        # 加载并处理DSM
        dsm = Image.open(DSM_FOLDER.format(id))
        dsm = dsm.resize((int(dsm.width * scale), int(dsm.height * scale)), Image.BILINEAR)
        
        # 转换为数组
        img_arr = np.array(img, dtype=np.float32) / 255.0
        lbl_arr = np.array(lbl, dtype=np.uint8)
        dsm_arr = np.array(dsm, dtype=np.float32)
        
        # 添加数据完整性检查
        if np.isnan(img_arr).any() or np.isinf(img_arr).any():
            print(f"警告: 图像 {id} 包含 NaN/Inf 值")
            img_arr = np.nan_to_num(img_arr)
        
        if np.isnan(dsm_arr).any() or np.isinf(dsm_arr).any():
            print(f"警告: DSM {id} 包含 NaN/Inf 值")
            dsm_arr = np.nan_to_num(dsm_arr)
        
        # 确保数据范围正确
        if np.min(img_arr) < 0 or np.max(img_arr) > 1.0:
            print(f"警告: 图像 {id} 超出 [0,1] 范围 ({np.min(img_arr)}, {np.max(img_arr)})")
            img_arr = np.clip(img_arr, 0, 1)
        
        # 归一化DSM
        dsm_arr = normalize_dsm(dsm_arr)
        
        # 随机裁剪带上下文的区域
        h, w = img_arr.shape[:2]
        crop_size = WINDOW_SIZE[0] + 2 * CONTEXT_MARGIN
        
        # 确保裁剪区域在图像范围内
        if w < crop_size or h < crop_size:
            # 图像太小，无法裁剪带上下文的区域
            # 使用整个图像并调整大小
            img_patch = img_arr
            lbl_patch = lbl_arr
            dsm_patch = dsm_arr
        else:
            x = random.randint(0, max(0, w - crop_size))
            y = random.randint(0, max(0, h - crop_size))
            
            # 提取带上下文的图像块
            img_patch = img_arr[y:y+crop_size, x:x+crop_size]
            lbl_patch = lbl_arr[y:y+crop_size, x:x+crop_size]
            dsm_patch = dsm_arr[y:y+crop_size, x:x+crop_size]
        
        # 转换为PIL图像进行大小调整
        img_patch_pil = Image.fromarray((img_patch * 255).astype(np.uint8))
        dsm_patch_pil = Image.fromarray((dsm_patch * 255).astype(np.uint8))
        lbl_patch_pil = Image.fromarray(lbl_patch.astype(np.uint8))
        
        # 调整到模型输入大小
        img_patch_resized = img_patch_pil.resize(WINDOW_SIZE, Image.BILINEAR)
        dsm_patch_resized = dsm_patch_pil.resize(WINDOW_SIZE, Image.BILINEAR)
        lbl_patch_resized = lbl_patch_pil.resize(WINDOW_SIZE, Image.NEAREST)
        
        # 转换为张量
        img_tensor = self.transform(img_patch_resized)
        lbl_tensor = torch.tensor(np.array(lbl_patch_resized), dtype=torch.long)
        dsm_tensor = torch.tensor(np.array(dsm_patch_resized), dtype=torch.float32).unsqueeze(0)
        
        return img_tensor, dsm_tensor, lbl_tensor

def find_and_write_filenames(directory, search_string):
    """
    遍历目录，查找包含指定字符串的JPG文件，并将文件名写入文本文件
    
    参数:
        directory: 要遍历的目录路径
        search_string: 要搜索的字符串
        output_file: 输出文本文件路径
    """
    # 收集符合条件的文件名（不含扩展名）
    matching_names = []
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查是否为JPG文件（不区分大小写）
        if filename.lower().endswith('.jpg'):
            # 检查文件名是否包含搜索字符串
            if search_string in os.path.splitext(filename)[0]:
                # 添加到结果列表（去掉扩展名）
                matching_names.append(os.path.splitext(filename)[0])
    #print(f"找到 {len(matching_names)} 个匹配文件")     
    
    return matching_names

# 主程序
if __name__ == "__main__":
    """
    DATA_FOLDER1 = "/root/autodl-tmp/dataset/cropped_Image"
    tem = []

    for i in tqdm(train_ids, desc="读取train_ids中……", leave=False):
        matching_names = find_and_write_filenames(DATA_FOLDER1, i)
        for j in [k for k in matching_names]:
            tem.append(j)
    train_ids = tem.copy() 
    
    tem = []
    for i in tqdm(test_ids, desc="读取test_ids中……", leave=False):
        matching_names = find_and_write_filenames(DATA_FOLDER1, i)
        for j in [k for k in matching_names]:
            tem.append(j)
    test_ids = tem

    with open('train.txt', 'w') as f:
        f.write('\n'.join(train_ids))

    with open('test.txt', 'w') as f:
        f.write('\n'.join(test_ids))
    """
    # 创建数据集和数据加载器
    train_set = EnhancedDataset(train_ids, scales=[0.75, 1.0, 1.25])
    test_set = EnhancedDataset(test_ids, scales=[1.0])  # 测试时使用单一尺度
    
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
    #train(net, optimizer, epochs=EPOCHS, scheduler=scheduler)
    end_time = time.time()
    print(f'总耗时: {(end_time - start_time)/60:.2f} 分钟')
    
    # 加载最佳模型进行最终测试
    best_model_path = max(glob('./experiment6/best_model_*.pth'), key=os.path.getctime)
    net.load_state_dict(torch.load(best_model_path))
    
    # 最终测试
    net.eval()
    pa, cpa, mpa, mIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
    print(f"最终测试pa: {pa:.4f}")
    print(f"最终测试cpa: {cpa}")
    print(f"最终测试mpa: {mpa:.4f}")
    print(f"最终测试mIoU: {mIoU:.4f}")
    
    # 可视化结果
    for i, (pred, id) in enumerate(zip(all_preds, test_ids)):
        if i < 100:  # 只可视化前5个结果
            plt.figure(figsize=(12, 4))
            
            # 原始图像
            plt.subplot(1, 3, 1)
            img = np.array(Image.open(DATA_FOLDER.format(id)))
            plt.imshow(img)
            plt.title('origin')
            
            # 真实标签
            plt.subplot(1, 3, 2)
            gt = np.array(Image.open(LABEL_FOLDER.format(id)))
            plt.imshow(gt)
            plt.title('ground truth')
            
            # 预测结果
            plt.subplot(1, 3, 3)
            plt.imshow(pred)
            plt.title('prediction')
            experiment
            plt.tight_layout()
            plt.savefig(f'./experiment6/result_{id}.png')
            plt.show()
        else:
            break
            
    with open("test_results.txt", "w") as f:
        f.write(f"最终测试pa: {pa:.4f}\n")
        f.write(f"最终测试cpa: {cpa}\n")
        f.write(f"最终测试mpa: {mpa:.4f}\n")
        f.write(f"最终测试mIoU: {mIoU:.4f}\n")