import random
import numpy as np
import torch

class PromptSimulator:
    """从GT自动生成训练提示
    论文参数:
      - mask概率: 0.5
      - point概率: 0.25
      - box概率: 0.25
      - num_frames_to_correct_for_train: 2 (最多2帧接收提示)
    """

    def __init__(self,
                 mask_prob=0.5,       # 论文: 0.5
                 point_prob=0.25,     # 论文: 0.25
                 box_prob=0.25,       # 论文: 0.25
                 num_pos_points=1,
                 num_neg_points=1,
                 box_jitter_ratio=0.1,
                 mask_noise_ratio=0.05):
        self.probs = [mask_prob, point_prob, box_prob]
        self.types = ['mask', 'point', 'box']
        self.num_pos_points = num_pos_points
        self.num_neg_points = num_neg_points
        self.box_jitter_ratio = box_jitter_ratio
        self.mask_noise_ratio = mask_noise_ratio

    def generate(self, gt_mask_3d):
        """
        gt_mask_3d: (N,) bool tensor, 3D点云GT掩码
        返回: dict with 'type' and 'data'
        """
        prompt_type = random.choices(self.types, weights=self.probs, k=1)[0]

        if prompt_type == 'mask':
            return self._gen_mask_prompt(gt_mask_3d)
        elif prompt_type == 'point':
            return self._gen_point_prompt(gt_mask_3d)
        else:
            return self._gen_box_prompt(gt_mask_3d)

    def _gen_mask_prompt(self, gt_mask_3d):
        """从GT mask加噪声生成mask提示"""
        noisy_mask = gt_mask_3d.clone().float()
        if self.mask_noise_ratio > 0:
            num_pts = gt_mask_3d.shape[0]
            num_flip = int(num_pts * self.mask_noise_ratio)
            flip_idx = torch.randperm(num_pts)[:num_flip]
            noisy_mask[flip_idx] = 1.0 - noisy_mask[flip_idx]
        return {'type': 'mask', 'mask': noisy_mask}

    def _gen_point_prompt(self, gt_mask_3d):
        """从GT mask采样正/负点"""
        pos_indices = torch.where(gt_mask_3d)[0]
        neg_indices = torch.where(~gt_mask_3d)[0]

        # 采样正点
        if len(pos_indices) > 0:
            sel = pos_indices[torch.randperm(len(pos_indices))[:self.num_pos_points]]
        else:
            sel = torch.tensor([], dtype=torch.long)

        # 采样负点
        if len(neg_indices) > 0:
            neg_sel = neg_indices[torch.randperm(len(neg_indices))[:self.num_neg_points]]
        else:
            neg_sel = torch.tensor([], dtype=torch.long)

        return {
            'type': 'point',
            'pos_indices': sel,
            'neg_indices': neg_sel,
            'labels': torch.cat([
                torch.ones(len(sel), dtype=torch.int32),
                torch.zeros(len(neg_sel), dtype=torch.int32)
            ])
        }

    def _gen_box_prompt(self, gt_mask_3d, points_xyz=None):
        """从GT mask对应的点云生成3D bounding box提示"""
        pos_indices = torch.where(gt_mask_3d)[0]
        if len(pos_indices) == 0:
            return self._gen_point_prompt(gt_mask_3d)  # fallback

        pos_points = points_xyz[pos_indices]  # (M, 3)
        min_coords = pos_points.min(dim=0).values
        max_coords = pos_points.max(dim=0).values

        # 加jitter
        extent = max_coords - min_coords
        jitter = (torch.rand(3) * 2 - 1) * extent * self.box_jitter_ratio
        min_coords = min_coords - jitter.abs()
        max_coords = max_coords + jitter.abs()

        return {
            'type': 'box',
            'min_coords': min_coords,
            'max_coords': max_coords,
        }
