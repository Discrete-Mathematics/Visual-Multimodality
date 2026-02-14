import os
import numpy as np
import torch
from PIL import Image
import json
import cv2
from torchvision import transforms

# RLE解码工具（COCO格式）
def rle_decode(rle, shape):
    from pycocotools import mask as maskUtils
    if isinstance(rle['counts'], list):
        return maskUtils.decode(rle)
    else:
        return maskUtils.decode({'counts': rle['counts'].encode('utf-8'), 'size': rle['size']})

def compose(frame_info):
    """
    frame_info: dict, 包含 pcd_path, label_path, image_paths (dict), frame_id, segment 等
    返回: dict, 包含 image, points, gt_instances
    """
    # 1. 加载图像（只用FRONT视角，可扩展多视角）
    img_path = frame_info['image_paths']['FRONT']
    img = Image.open(img_path).convert('RGB')
    # resize到[3,768,768]，归一化到[0,1]
    img_trans = transforms.Compose([
        transforms.Resize((768, 768)),
        transforms.ToTensor(),
    ])
    img_tensor = img_trans(img)  # [3,768,768]

    # 2. 加载点云
    pcd = np.load(frame_info['pcd_path'])
    points = pcd['data']  # [N,3]
    # 固定点数（如65536），不足补零，超出随机采样
    max_points = 65536
    N = points.shape[0]
    if N >= max_points:
        idx = np.random.choice(N, max_points, replace=False)
        points = points[idx]
    else:
        pad = np.zeros((max_points - N, 3), dtype=points.dtype)
        points = np.concatenate([points, pad], axis=0)
    points_tensor = torch.from_numpy(points).float()  # [65536,3]

    # 3. 加载标签
    with open(frame_info['label_path'], 'r') as f:
        label = json.load(f)
    # 只用 FRONT 相机的 objects
    objects = label['sensors']['FRONT']['objects']
    gt_instances = []
    for obj in objects:
        # 解码mask
        mask = rle_decode(obj['image_mask'], obj['image_mask']['size'])  # [H,W], 0/1
        # resize到[768,768]
        mask = cv2.resize(mask.astype(np.uint8), (768, 768), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()  # [1,768,768]
        gt_instances.append({
            'object_id': obj['object_id'],
            'score': obj.get('score', 1.0),
            'volume': obj.get('volume', 0),
            'iou': obj.get('iou', 1.0),
            'semantic': obj.get('semantic', 'unknown'),
            'mask': mask_tensor,
        })

    return {
        'image': img_tensor,         # [3,768,768] float32
        'points': points_tensor,     # [65536,3] float32
        'gt_instances': gt_instances # list of dict
    }