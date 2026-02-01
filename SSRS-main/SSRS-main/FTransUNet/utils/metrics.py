import torch
import torch.nn.functional as F
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def normalize_tensor(tensor):
    """将张量归一化到[0,1]范围，用于指标计算"""
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val - min_val < 1e-6:
        return torch.zeros_like(tensor)
    return (tensor - min_val) / (max_val - min_val)

# ===================== 无监督指标（深度补全/图像增强） =====================
def calculate_ssim(tensor1, tensor2):
    """
    计算SSIM（结构相似性），无监督，无需GT
    tensor1: 输入图（如残缺深度图），shape [B, C, H, W]
    tensor2: 输出图（如补全深度图），shape [B, C, H, W]
    返回：批次平均SSIM值
    """
    assert tensor1.shape == tensor2.shape, "两张图的形状必须一致"
    tensor1 = normalize_tensor(tensor1).cpu().detach().numpy()
    tensor2 = normalize_tensor(tensor2).cpu().detach().numpy()
    
    ssim_scores = []
    for b in range(tensor1.shape[0]):
        # 单张图处理：[C, H, W] → [H, W]（灰度图）
        img1 = tensor1[b].squeeze()  # 去除通道维（C=1）
        img2 = tensor2[b].squeeze()
        
        # 计算SSIM
        score = ssim(img1, img2, data_range=1.0)
        ssim_scores.append(score)
    
    return np.mean(ssim_scores)

def calculate_gradient_similarity(tensor1, tensor2):
    """
    计算梯度相似度（Grad SIM），无监督，无需GT
    基于Sobel算子计算梯度，再衡量梯度图的余弦相似度
    """
    assert tensor1.shape == tensor2.shape, "两张图的形状必须一致"
    tensor1 = normalize_tensor(tensor1).cpu().detach().numpy()
    tensor2 = normalize_tensor(tensor2).cpu().detach().numpy()
    
    grad_scores = []
    for b in range(tensor1.shape[0]):
        img1 = tensor1[b].squeeze() * 255
        img2 = tensor2[b].squeeze() * 255
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        
        # 计算Sobel梯度
        sobel_x1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
        grad1 = np.sqrt(sobel_x1**2 + sobel_y1**2)
        
        sobel_x2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
        grad2 = np.sqrt(sobel_x2**2 + sobel_y2**2)
        
        # 归一化梯度图
        grad1 = (grad1 - grad1.min()) / (grad1.max() - grad1.min() + 1e-6)
        grad2 = (grad2 - grad2.min()) / (grad2.max() - grad2.min() + 1e-6)
        
        # 计算余弦相似度
        flat1 = grad1.flatten()
        flat2 = grad2.flatten()
        cos_sim = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2) + 1e-6)
        grad_scores.append(cos_sim)
    
    return np.mean(grad_scores)

def calculate_phash_similarity(tensor1, tensor2):
    """
    计算感知哈希相似度，无监督，无需GT
    基于pHash算法，衡量整体结构一致性
    """
    assert tensor1.shape == tensor2.shape, "两张图的形状必须一致"
    tensor1 = normalize_tensor(tensor1).cpu().detach().numpy()
    tensor2 = normalize_tensor(tensor2).cpu().detach().numpy()
    
    phash_scores = []
    for b in range(tensor1.shape[0]):
        img1 = tensor1[b].squeeze() * 255
        img2 = tensor2[b].squeeze() * 255
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        
        # 生成pHash
        def get_phash(img):
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dct = cv2.dct(np.float32(img_gray))
            dct_roi = dct[:8, :8]
            avg = np.mean(dct_roi[1:])
            hash_val = (dct_roi > avg).flatten()
            return hash_val
        
        hash1 = get_phash(img1)
        hash2 = get_phash(img2)
        
        # 计算汉明距离，转换为相似度
        hamming_dist = np.sum(hash1 != hash2)
        similarity = 1 - (hamming_dist / len(hash1))
        phash_scores.append(similarity)
    
    return np.mean(phash_scores)

# ===================== 有监督指标（语义分割） =====================
def calculate_pixel_accuracy(pred_seg, seg_gt, ignore_index=-1):
    """
    计算像素准确率（Pixel Acc）
    pred_seg: [B, C, H, W] → 需先转为argmax结果 [B, H, W]
    seg_gt: [B, H, W]
    """
    # 转换预测结果
    pred_seg = torch.argmax(pred_seg, dim=1).cpu().detach().numpy()
    seg_gt = seg_gt.cpu().detach().numpy()
    
    # 忽略指定索引
    mask = (seg_gt != ignore_index)
    correct = np.sum((pred_seg[mask] == seg_gt[mask]))
    total = np.sum(mask)
    
    return correct / total if total > 0 else 0.0

def calculate_miou(pred_seg, seg_gt, num_classes=19, ignore_index=-1):
    """
    计算平均交并比（mIoU）
    """
    pred_seg = torch.argmax(pred_seg, dim=1).cpu().detach().numpy()
    seg_gt = seg_gt.cpu().detach().numpy()
    
    iou_scores = []
    for cls in range(num_classes):
        # 忽略背景/无效索引
        cls_mask_gt = (seg_gt == cls) & (seg_gt != ignore_index)
        cls_mask_pred = (pred_seg == cls) & (seg_gt != ignore_index)
        
        if np.sum(cls_mask_gt) == 0 and np.sum(cls_mask_pred) == 0:
            continue  # 跳过无样本的类别
        
        # 计算交并比
        intersection = np.sum(cls_mask_gt & cls_mask_pred)
        union = np.sum(cls_mask_gt | cls_mask_pred)
        iou = intersection / (union + 1e-6)
        iou_scores.append(iou)
    
    return np.mean(iou_scores) if len(iou_scores) > 0 else 0.0