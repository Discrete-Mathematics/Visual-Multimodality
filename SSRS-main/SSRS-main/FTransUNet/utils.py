import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import itertools
from torchvision.utils import make_grid
from PIL import Image
from skimage import io
import os
import os
import sys
from pathlib import Path
from metric import *
import torch.nn as nn
def initialize_weights(model):
    for name, param in model.named_parameters():
        if param.dim() > 1 and 'weight' in name:
            nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        elif 'bias' in name:
            nn.init.constant_(param, 0)
    print('✅ 模型权重初始化完成')

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
    """绘制训练历史图表"""
    if not train_history:
        return
    
    epochs = [entry['epoch'] for entry in train_history]
    train_losses = [entry['train_loss'] for entry in train_history]
    val_mIoUs = [entry['val_metrics']['mIoU'] for entry in train_history]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='train loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('train loss curve')
    ax1.legend()
    ax1.grid(True)
    
    # mIoU曲线
    ax2.plot(epochs, val_mIoUs, 'r-', label='val mIoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU')
    ax2.set_title('val mIoU curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 训练历史图表已保存")

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

# 修改数据集路径配置
DATA_FOLDER = "/root/autodl-tmp/dataset/croped_Image/{}.jpg"
LABEL_FOLDER = "/root/autodl-tmp/dataset/croped_Label/{}_label.png"
DSM_FOLDER = "/root/autodl-tmp/dataset/croped_Depth/{}.png"
N_CLASSES = 16
BATCH_SIZE = 10  # 根据GPU内存调整
Stride_Size = 128
WINDOW_SIZE = (256, 256)

def get_file_list(directory, recursive=False, include_ext=True, sort=True):
    """
    获取目录下的所有文件名列表
    
    参数:
    directory: 目标目录路径
    recursive: 是否递归获取子目录中的文件 (默认: False)
    include_ext: 是否包含文件扩展名 (默认: True)
    sort: 是否对结果进行排序 (默认: True)
    
    返回:
    文件名列表
    """
    file_list = []
    
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"错误: 目录 '{directory}' 不存在")
        return []
    
    # 检查是否是目录
    if not os.path.isdir(directory):
        print(f"错误: '{directory}' 不是一个目录")
        return []
    
    # 递归获取所有文件
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                # 获取相对路径
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                
                # 处理是否包含扩展名
                if not include_ext:
                    rel_path = os.path.splitext(rel_path)[0]
                
                file_list.append(rel_path)
    
    # 非递归获取文件
    else:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            
            # 只添加文件，跳过目录
            if os.path.isfile(item_path):
                # 处理是否包含扩展名
                if not include_ext:
                    item = os.path.splitext(item)[0]
                
                file_list.append(item)
    
    # 排序
    if sort:
        file_list.sort()
    
    return file_list

def save_as_python_list(file_list, output_file=None):
    """
    将文件列表保存为Python列表格式
    
    参数:
    file_list: 文件列表
    output_file: 输出文件路径 (如果为None，则打印到控制台)
    """
    # 创建Python列表字符串
    list_str = "file_list = [\n"
    for file in file_list:
        list_str += f"    '{file}',\n"
    list_str += "]"
    
    # 输出结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(list_str)
        print(f"文件列表已保存到: {output_file}")
    else:
        print("\n文件列表 (Python格式):")
        print(list_str)

# Parameters
## SwinFusion
# WINDOW_SIZE = (64, 64) # Patch size
"""
WINDOW_SIZE = (256, 256) # Patch size

STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
FOLDER = "/root/autodl-tmp/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10 # Number of samples in a mini-batch

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory

train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
test_ids = ['5', '21', '15', '30']
DATASET = 'Vaihingen'
Stride_Size = 32
MAIN_FOLDER = FOLDER + 'Vaihingen/ISPRS_semantic_labeling_Vaihingen/'
DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_09cm_matching_area{}.tif'
LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
ERODED_FOLDER = MAIN_FOLDER + 'ISPRS_semantic_labeing_Vaihingen_ground_truth_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'

# train_ids = ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12', '5_12', '7_11', '7_9', '6_9', '7_7',
#              '4_12', '6_8', '6_12', '6_7', '4_11']
# test_ids = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']
# DATASET = 'Potsdam'
# Stride_Size = 128 # for quickly training
# MAIN_FOLDER = FOLDER + 'Potsdam/'
# DATA_FOLDER = MAIN_FOLDER + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
# DSM_FOLDER = MAIN_FOLDER + '1_DSM_normalisation/dsm_potsdam_{}_normalized_lastools.jpg'
# LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
# ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'
"""
# ISPRS color palette
# Let's define the standard ISPRS color palette
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def save_img(tensor, name):
    tensor = tensor.cpu() .permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')

class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        # self.boundary_files = [BOUNDARY_FOLDER.format(id) for id in ids]
        self.dsm_files = [DSM_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        # self.boundary_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        # Default epoch size is 10 000 samples
        if DATASET == 'Potsdam':
            return BATCH_SIZE * 1000
        elif DATASET == 'Vaihingen':
            return BATCH_SIZE * 1000
        else:
            return None

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            ## Potsdam IRRG
            if DATASET == 'Potsdam':
                ## RGB
                data = io.imread(self.data_files[random_idx])[:, :, :3].transpose((2, 0, 1))
                ## IRRG
                # data = io.imread(self.data_files[random_idx])[:, :, (3, 0, 1, 2)][:, :, :3].transpose((2, 0, 1))
                data = 1 / 255 * np.asarray(data, dtype='float32')
            else:
            ## Vaihingen IRRG
                data = io.imread(self.data_files[random_idx])
                data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data
        
        # if random_idx in self.boundary_cache_.keys():
        #     boundary = self.boundary_cache_[random_idx]
        # else:
        #     boundary = np.asarray(io.imread(self.boundary_files[random_idx])) / 255
        #     boundary = boundary.astype(np.int64)
        #     if self.cache:
        #         self.boundary_cache_[random_idx] = boundary

        if random_idx in self.dsm_cache_.keys():
            dsm = self.dsm_cache_[random_idx]
        else:
            # DSM is normalized in [0, 1]
            dsm = np.asarray(io.imread(self.dsm_files[random_idx]), dtype='float32')
            min = np.min(dsm)
            max = np.max(dsm)
            dsm = (dsm - min) / (max - min)
            if self.cache:
                self.dsm_cache_[random_idx] = dsm

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        dsm_p = dsm[x1:x2, y1:y2]
        # boundary_p = boundary[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        # data_p, boundary_p, label_p = self.data_augmentation(data_p, boundary_p, label_p)
        data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(dsm_p),
                torch.from_numpy(label_p))
        
## We load one tile from the dataset and we display it
# img = io.imread('./ISPRS_dataset/Vaihingen/top/top_mosaic_09cm_area11.tif')
# fig = plt.figure()
# fig.add_subplot(121)
# plt.imshow(img)
#
# # We load the ground truth
# gt = io.imread('./ISPRS_dataset/Vaihingen/gts_for_participants/top_mosaic_09cm_area11.tif')
# fig.add_subplot(122)
# plt.imshow(gt)
# plt.show()
#
# # We also check that we can convert the ground truth into an array format
# array_gt = convert_from_color(gt)
# print("Ground truth in numerical format has shape ({},{}) : \n".format(*array_gt.shape[:2]), array_gt)


# Utils

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

"""
def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size
"""


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

"""
def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

    print("Confusion matrix :")
    print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" %(kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[:5])
    print('mean MIoU: %.4f' % (MIoU))
    print("---")

    return accuracy
"""

def accuracy(pred, gt):
    """计算像素精度"""
    valid = (gt != 255)  # 忽略特定值（如边界）
    acc = (pred[valid] == gt[valid]).sum() / valid.sum()
    return acc.item() * 100

def metrics(preds, gts):
    """综合评估函数，可替换为您的评估指标"""
    metric = SegmentationMetric(N_CLASSES)
    metric.addBatch(preds, gts)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    
    return pa, cpa, mpa, mIoU
    

if __name__ == "__main__":
    # 在这里设置目标目录
    TARGET_DIR = "/root/MUSeg/dataset/Depth"  # 替换为您的目录路径
    
    # 设置选项
    RECURSIVE = False      # 是否递归获取子目录中的文件
    INCLUDE_EXT = True     # 是否包含文件扩展名
    SORT = True            # 是否对结果排序
    OUTPUT_FILE = None     # 输出文件路径 (None表示打印到控制台)
    # OUTPUT_FILE = "file_list.py"  # 取消注释以保存到文件
    
    # 获取文件列表
    files = get_file_list(
        directory=TARGET_DIR,
        recursive=RECURSIVE,
        include_ext=INCLUDE_EXT,
        sort=SORT
    )
    
    # 输出结果
    if files:
        print(f"找到 {len(files)} 个文件:")
        save_as_python_list(files, OUTPUT_FILE)
    else:
        print("目录中没有找到文件")
