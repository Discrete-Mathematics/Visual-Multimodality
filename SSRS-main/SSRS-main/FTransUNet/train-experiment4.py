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
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import json

def count_sliding_window(image, step, window_size):
    # 获取图像尺寸
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    elif isinstance(image, Image.Image):
        w, h = image.size
    else:
        raise TypeError("不支持的图像类型")
    
    # 处理步长参数
    if isinstance(step, int):
        step_x = step_y = step
    else:
        step_y, step_x = step
    
    # 获取窗口尺寸
    win_h, win_w = window_size
    
    # 计算高度方向的窗口数
    n_y = (h - win_h) // step_y + 1
    if (h - win_h) % step_y != 0:
        n_y += 1  # 添加一行以覆盖剩余部分
    
    # 计算宽度方向的窗口数
    n_x = (w - win_w) // step_x + 1
    if (w - win_w) % step_x != 0:
        n_x += 1  # 添加一列以覆盖剩余部分
    
    return n_y * n_x

def pil_to_tensor_no_normalize(pil_image):
    """
    将 PIL Image 转换为 PyTorch 张量，保持原始像素值 (0-255)
    支持所有 PIL 图像类型
    
    参数:
        pil_image: PIL.Image 对象 (任何子类)
    
    返回:
        torch.Tensor: 形状为 (C, H, W)，数据类型为 uint8
    """
    # 转换为 NumPy 数组
    np_array = np.array(pil_image)
    
    # 处理通道顺序
    if np_array.ndim == 2:  # 灰度图 (H, W)
        np_array = np.expand_dims(np_array, axis=0)  # 添加通道维度 -> (1, H, W)
    else:  # 彩色图 (H, W, C)
        np_array = np_array.transpose(2, 0, 1)  # (C, H, W)
    
    # 转换为 PyTorch 张量
    tensor = torch.from_numpy(np_array).float()
    
    return tensor

def gt_to_tensor_no_normalize(pil_image):
    """
    将 PIL Image 转换为 PyTorch 张量，保持原始像素值 (0-255)
    支持所有 PIL 图像类型
    
    参数:
        pil_image: PIL.Image 对象 (任何子类)
    
    返回:
        torch.Tensor: 形状为 (C, H, W)，数据类型为 uint8
    """
    # 转换为 NumPy 数组
    np_array = np.array(pil_image)
    
    # 处理通道顺序
    if np_array.ndim == 2:  # 灰度图 (H, W)
        np_array = np.expand_dims(np_array, axis=0)  # 添加通道维度 -> (1, H, W)
    else:  # 彩色图 (H, W, C)
        np_array = np_array.transpose(2, 0, 1)  # (C, H, W)
    
    # 转换为 PyTorch 张量
    tensor = torch.from_numpy(np_array).long()
    
    return tensor

def random_mirror(rgb, gt, modal_x):
    """
    使用 PIL 实现的随机镜像函数
    """
    if np.random.rand() > 0.5:
        # 水平翻转
        rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        modal_x = modal_x.transpose(Image.FLIP_LEFT_RIGHT)
    
    return rgb, gt, modal_x

def random_scale(rgb, gt, modal_x, scale_array):
    """
    使用 PIL 实现的随机缩放函数
    """
    # 从缩放比例数组中随机选择一个比例
    scale = np.random.choice(scale_array)
    
    # 计算新尺寸
    w, h = rgb.size
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图像
    rgb = rgb.resize((new_w, new_h), Image.BILINEAR)
    gt = gt.resize((new_w, new_h), Image.NEAREST)
    modal_x = modal_x.resize((new_w, new_h), Image.BILINEAR)
    
    return rgb, gt, modal_x, scale

def normalize(image, mean, std):
    """
    标准化函数 - 支持 PIL Image 类型
    """
    # 创建转换流程
    transform = transforms.Compose([
        #transforms.ToTensor(),  # 转换为张量 (0-1范围)
        pil_to_tensor_no_normalize,
        transforms.Normalize(mean=mean, std=std)  # 标准化
    ])
    
    # 应用转换
    return transform(image)
    
class Config:
    def __init__(self):
        self.train_scale_array = [0.5, 0.75, 1.0, 1.25, 1.5]  # 随机缩放比例
        self.image_height = 1536  # 目标高度
        self.image_width = 2048   # 目标宽度
        self.use_resize_instead_of_crop = True  # 使用resize+pad模式
        self.resize_size = (256, 256)  # resize目标尺寸

config = Config()

class TrainPre(object):
    def __init__(self, norm_mean=[59.267183, 58.158282, 70.550627], norm_std=[53.042690, 56.694345, 57.744167], sign=True, config=config):
        self.config = config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign = sign

    def __call__(self, rgb, gt, modal_x):
        # 应用随机镜像和缩放
        rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        if self.config.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(rgb, gt, modal_x, self.config.train_scale_array)
        
        # 使用PIL的resize方法替代cv2.resize
        target_h, target_w = self.config.resize_size
        
        # 计算缩放比例，保持长宽比
        w, h = rgb.size  # PIL Image使用.size属性
        scale = min(target_w / w, target_h / h)
        
        # 计算新尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # resize图像 - 使用PIL方法
        rgb = rgb.resize((new_w, new_h), Image.BILINEAR)
        gt = gt.resize((new_w, new_h), Image.NEAREST)
        modal_x = modal_x.resize((new_w, new_h), Image.BILINEAR)
        
        # 如果resize后的尺寸小于目标尺寸，进行填充
        if new_w < target_w or new_h < target_h:
            # 创建新图像（居中放置）
            new_rgb = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            new_gt = Image.new("L", (target_w, target_h), 0)  # 忽略值0
            new_modal_x = Image.new("L", (target_w, target_h), 0)  # 深度图填充0
            
            # 计算位置（居中）
            left = (target_w - new_w) // 2
            top = (target_h - new_h) // 2
            
            # 粘贴图像
            new_rgb.paste(rgb, (left, top))
            new_gt.paste(gt, (left, top))
            new_modal_x.paste(modal_x, (left, top))
            
            rgb, gt, modal_x = new_rgb, new_gt, new_modal_x
        
        # 应用标准化
        rgb_tensor = normalize(rgb, self.norm_mean, self.norm_std)
        
        if self.sign:
            modal_x_tensor = normalize(modal_x, [0.48], [0.28])
        else:
            modal_x_tensor = normalize(modal_x, self.norm_mean, self.norm_std)
        
        # 标签图只需转换为张量
        gt_tensor = gt_to_tensor_no_normalize(gt).squeeze()
        
        return rgb_tensor, gt_tensor, modal_x_tensor
    
os.makedirs('./experiment5', exist_ok=True)

# 优化PyTorch显存分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 环境检查
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"cuDNN版本: {torch.backends.cudnn.version()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 修改数据集路径配置
DATA_FOLDER = "/root/autodl-tmp/dataset/Image/{}.jpg"
LABEL_FOLDER = "/root/autodl-tmp/dataset/Label_Label/{}_label.png"
DSM_FOLDER = "/root/autodl-tmp/dataset/Depth/{}.png"
N_CLASSES = 16
BATCH_SIZE = 32
EPOCHS = 500
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
                
                #target = target.squeeze(1)
                #print(target)
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
                torch.save(net.state_dict(), f'./experiment5/best_model_epoch{epoch}_mIoU{mIoU:.4f}.pth')
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
        #scale = random.choice(self.scales)
        
        # 加载图像
        img = Image.open(DATA_FOLDER.format(id)).convert('RGB')
        #img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
        
        # 加载标签
        lbl = Image.open(LABEL_FOLDER.format(id))
        #lbl = lbl.resize((int(lbl.width * scale), int(lbl.height * scale)), Image.NEAREST)
        
        # 加载并处理DSM
        dsm = Image.open(DSM_FOLDER.format(id))
        #dsm = dsm.resize((int(dsm.width * scale), int(dsm.height * scale)), Image.BILINEAR)

        train_pre = TrainPre()
        img_tensor, lbl_tensor, dsm_tensor = train_pre(img, lbl, dsm)
        
        return img_tensor, dsm_tensor, lbl_tensor

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

# 主程序
if __name__ == "__main__":
    # # 获取文件列表
    # TARGET_DIR = "/root/autodl-tmp/dataset/Depth"
    # all_ids = [os.path.splitext(f)[0] for f in os.listdir(TARGET_DIR) if f.endswith('.png')]
    
    # # 划分训练集和测试集
    # #train_size = int(0.8 * len(all_ids))
    # test_size = len(all_ids) - train_size
    # train_ids, test_ids = torch.utils.data.random_split(all_ids, [train_size, test_size])
    train_ids = read_txt_to_list("train.txt")
    test_ids = read_txt_to_list("val.txt")
    print(f"训练集大小: {len(train_ids)}")
    print(f"测试集大小: {len(test_ids)}")
    
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
    best_model_path = max(glob('./experiment5/best_model_*.pth'), key=os.path.getctime)
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
        if i < 5:  # 只可视化前5个结果
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
            
            plt.tight_layout()
            plt.savefig(f'./experiment5/result_{id}.png')
            plt.show()

    
    results = {
        "pa": round(pa, 4),
        "cpa": cpa,  # cpa可能是列表/数组，直接保留原始结构
        "mpa": round(mpa, 4),
        "mIoU": round(mIoU, 4)
    }
    
    # 写入JSON文件
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)