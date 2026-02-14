# augmentations.py
# SAM4D 训练数据增强模块

import random
import numpy as np
import torch
import torchvision.transforms as T


def rotate_z(points, angle_deg):
    """绕Z轴旋转点云"""
    angle = np.radians(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    rot = torch.tensor([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]],
                        dtype=points.dtype, device=points.device)
    return points @ rot.T


class SequenceAugmentation:
    """序列级增强 - 对所有帧一致应用
    论文参数 (Table A1):
      - hflip
      - affine degrees: 25
      - LiDAR rotation-Z: 45°
      - LiDAR hflip, vflip
    """

    def __init__(self,
                 hflip_prob=0.5,
                 affine_degrees=25,        # 论文: 25°
                 lidar_rot_z_degrees=45,   # 论文: 45°
                 lidar_hflip_prob=0.5,
                 lidar_vflip_prob=0.5):
        self.hflip_prob = hflip_prob
        self.affine_degrees = affine_degrees
        self.lidar_rot_z_degrees = lidar_rot_z_degrees
        self.lidar_hflip_prob = lidar_hflip_prob
        self.lidar_vflip_prob = lidar_vflip_prob

    def __call__(self, frames):
        """对整个序列一致应用增强"""
        do_hflip = random.random() < self.hflip_prob
        rot_angle = random.uniform(-self.affine_degrees, self.affine_degrees)
        lidar_rot = random.uniform(-self.lidar_rot_z_degrees, self.lidar_rot_z_degrees)
        do_lidar_hflip = random.random() < self.lidar_hflip_prob
        do_lidar_vflip = random.random() < self.lidar_vflip_prob

        for frame in frames:
            # 图像增强
            if do_hflip:
                frame['image'] = frame['image'].flip(-1)
                if 'gt_masks_2d' in frame:
                    frame['gt_masks_2d'] = frame['gt_masks_2d'].flip(-1)
            frame['image'] = T.functional.affine(
                frame['image'], angle=rot_angle, translate=[0, 0], scale=1.0, shear=0)

            # LiDAR增强
            if do_lidar_hflip:
                frame['points'][:, 1] *= -1
            if do_lidar_vflip:
                frame['points'][:, 0] *= -1
            frame['points'][:, :3] = rotate_z(frame['points'][:, :3], lidar_rot)

            # GT同步变换（bbox、3D mask等）
            if do_hflip and 'gt_boxes_2d' in frame:
                W = frame['image'].shape[-1]
                frame['gt_boxes_2d'][:, [0, 2]] = W - frame['gt_boxes_2d'][:, [2, 0]]
            if (do_lidar_hflip or do_lidar_vflip) and 'gt_boxes_3d' in frame:
                if do_lidar_hflip:
                    frame['gt_boxes_3d'][:, 1] *= -1
                    frame['gt_boxes_3d'][:, 6] *= -1  # yaw
                if do_lidar_vflip:
                    frame['gt_boxes_3d'][:, 0] *= -1
                    frame['gt_boxes_3d'][:, 6] = np.pi - frame['gt_boxes_3d'][:, 6]

        return frames


class FrameAugmentation:
    """帧级增强 - 每帧独立应用
    论文参数 (Table A1):
      - colorjitter: brightness=0.1, contrast=0.03, saturation=0.03
      - grayscale prob: 0.05
    """

    def __init__(self,
                 brightness=0.1,     # 论文: 0.1
                 contrast=0.03,      # 论文: 0.03
                 saturation=0.03,    # 论文: 0.03
                 grayscale_prob=0.05 # 论文: 0.05
                 ):
        self.color_jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation)
        self.grayscale = T.RandomGrayscale(p=grayscale_prob)

    def __call__(self, frame):
        frame['image'] = self.color_jitter(frame['image'])
        frame['image'] = self.grayscale(frame['image'])
        return frame


def build_augmentations(cfg):
    """根据配置构建数据增强组件

    Args:
        cfg: 包含 train_augmentation 子字段的配置dict
    Returns:
        dict with 'sequence' and 'frame' augmentation objects
    """
    aug_cfg = cfg.get('train_augmentation', {})

    seq_cfg = aug_cfg.get('sequence', {})
    sequence_aug = SequenceAugmentation(
        hflip_prob=seq_cfg.get('hflip_prob', 0.5),
        affine_degrees=seq_cfg.get('affine_degrees', 25),
        lidar_rot_z_degrees=seq_cfg.get('lidar_rot_z_degrees', 45),
        lidar_hflip_prob=seq_cfg.get('lidar_hflip_prob', 0.5),
        lidar_vflip_prob=seq_cfg.get('lidar_vflip_prob', 0.5),
    )

    frame_cfg = aug_cfg.get('frame', {})
    frame_aug = FrameAugmentation(
        brightness=frame_cfg.get('brightness', 0.1),
        contrast=frame_cfg.get('contrast', 0.03),
        saturation=frame_cfg.get('saturation', 0.03),
        grayscale_prob=frame_cfg.get('grayscale_prob', 0.05),
    )

    return {'sequence': sequence_aug, 'frame': frame_aug}
