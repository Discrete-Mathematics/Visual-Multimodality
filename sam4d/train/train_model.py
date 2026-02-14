import random
import torch.nn as nn

from ..sam4d_base import SAM4DBase
from .loss import SigmoidFocalLoss, DiceLoss
from .utils_train import filter_and_sample_objects, generate_occlusion_labels, compute_iou, accumulate_loss, custom_collate, mask_to_box_with_jitter, add_noise_to_mask, sample_points_from_mask
from .prompt import PromptSimulator
from .pipelines.augmentations import SequenceAugmentation, FrameAugmentation, build_augmentations

class SAM4DTrain(SAM4DBase):
    """SAM4D训练类"""
    def __init__(self, cfg):
        import torch.nn as nn
        from mmengine import MODELS
        import copy
        # 直接用cfg中的backbone_cfg、fusion_cfg、head_cfg
        backbone_cfg = getattr(cfg, 'backbone_cfg', None)
        fusion_cfg = getattr(cfg, 'fusion_cfg', None)
        head_cfg = getattr(cfg, 'head_cfg', None)
        # 自动补全decoder_channels长度，兼容MinkUNet实现
        if backbone_cfg is not None and 'point' in backbone_cfg:
            point_cfg = backbone_cfg['point']
            if isinstance(point_cfg, dict):
                for name, subcfg in point_cfg.items():
                    if isinstance(subcfg, dict) and 'minkunet' in name.lower():
                        dc = subcfg.get('decoder_channels', None)
                        if isinstance(dc, list) and len(dc) < 3:
                            # 自动补足decoder_channels
                            last = dc[-1] if len(dc) > 0 else 32
                            while len(dc) < 3:
                                dc.append(last // 2)
                            subcfg['decoder_channels'] = dc
                        elif dc is None:
                            # 若未设置，自动补全
                            subcfg['decoder_channels'] = [256, 128, 64]
        # 本地构造backbone
        backbone = nn.ModuleDict()
        if backbone_cfg is not None:
            if 'image' in backbone_cfg:
                image_cfg = backbone_cfg['image']
                if isinstance(image_cfg, list):
                    image_modules = [MODELS.build(subcfg) for subcfg in image_cfg]
                    backbone['image'] = nn.Sequential(*image_modules)
                elif isinstance(image_cfg, dict):
                    backbone['image'] = MODELS.build(image_cfg)
            if 'point' in backbone_cfg:
                point_cfg = backbone_cfg['point']
                if isinstance(point_cfg, dict):
                    backbone['point'] = nn.ModuleDict()
                    for name in point_cfg:
                        if isinstance(point_cfg[name], list):
                            backbone['point'][name] = nn.Sequential(*[MODELS.build(subcfg) for subcfg in point_cfg[name]])
                        elif isinstance(point_cfg[name], dict):
                            backbone['point'][name] = MODELS.build(point_cfg[name])
        # fusion和head
        def patch_use_rope_all(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k == 'self_attention' and isinstance(v, dict):
                        v.setdefault('use_rope', False)
                    patch_use_rope_all(v)
            elif isinstance(d, list):
                for item in d:
                    patch_use_rope_all(item)
        fusion_cfg_cp = copy.deepcopy(fusion_cfg)
        patch_use_rope_all(fusion_cfg_cp)
        from sam4d.model_template import build_fusion, build_head
        fusion = build_fusion(fusion_cfg_cp)
        head = build_head(head_cfg)
        # 只传递dict给super().__init__，不传递nn.Module
        model_cfg = {}
        # 你可以根据需要补充其他参数
        super().__init__(
            voxelization=None,
            backbone=backbone_cfg,
            fusion=fusion_cfg,
            head=head_cfg,
        )
        # 本地保存nn.ModuleDict
        self.local_backbone = backbone
        self.local_fusion = fusion
        self.local_head = head
        # 下面是本地nn.ModuleDict的实际用法（如需）
        # self.backbone = backbone
        # self.fusion = fusion
        # self.head = head
        # 你可以根据需要补充其他参数
        # 例如：
        #   feat_dim=model_cfg.get('feat_dim', 256),
        #   mem_dim=model_cfg.get('mem_dim', 256),
        #   ...
        # 1. 损失函数
        self.focal_loss = SigmoidFocalLoss()
        self.dice_loss = DiceLoss(weight=getattr(cfg, 'mask_dice_weight', 1.0))
        self.iou_loss = nn.L1Loss()
        self.occ_loss = nn.CrossEntropyLoss()
        # 2. 训练参数
        self.max_num_objects = getattr(cfg, 'max_num_objects', 6)
        self.num_frames = getattr(cfg, 'num_frames', 8)
        self.num_frames_to_correct = getattr(cfg, 'num_frames_to_correct_for_train', 2)
        # 3. 数据增强
        self.train_augmentations = build_augmentations(cfg)
    
    def forward(self, batch):
        frames = batch['frames']
        objects = batch['objects']

        # 序列级增强（所有帧一致）
        frames = self.sequence_aug(frames)
        # 帧级增强（每帧独立）
        for frame in frames:
            self.frame_aug(frame)

        self.memory_bank = {}
        total_loss = {}

        for obj in objects:
            obj_id = id(obj)   # 实际应用需使用唯一ID
            for t in range(self.num_frames):
                frame = frames[t]
                # 提取特征（需确保shape正确，通常添加batch维度）
                img_feat = self.image_encoder(frame['image'].unsqueeze(0))
                pts_feat = self.lidar_encoder(frame['points'].unsqueeze(0))
                fused_feat = self.fusion(img_feat, pts_feat)

                gt_mask = obj['mask_sequence'][t]
                gt_occ = obj['occ_sequence'][t]

                if t < self.num_frames_to_correct:
                    points_xyz = frame['points'][:, :3]   # 假设点云前3列为xyz
                    prompt = self._generate_prompt(gt_mask, points_xyz)
                    pred_mask, pred_iou, pred_occ = self.decode_with_prompt(fused_feat, prompt)
                else:
                    memory = self.memory_bank.get(obj_id)
                    if memory is None:
                        points_xyz = frame['points'][:, :3]
                        prompt = self._generate_prompt(gt_mask, points_xyz)
                        pred_mask, pred_iou, pred_occ = self.decode_with_prompt(fused_feat, prompt)
                    else:
                        pred_mask, pred_iou, pred_occ = self.decode_from_memory(fused_feat, memory)

                self._update_memory(obj_id, pred_mask.detach(), fused_feat.detach())

                loss = self._compute_loss(pred_mask, pred_iou, pred_occ, gt_mask, gt_occ)
                total_loss = accumulate_loss(total_loss, loss)

        return total_loss    
    def _compute_loss(self, pred_mask, pred_iou, pred_occ, gt_mask, gt_occ):
        loss_focal = self.focal_loss(pred_mask, gt_mask) * 20.0
        loss_dice = self.dice_loss(pred_mask, gt_mask) * 1.0
        loss_iou = self.iou_loss(pred_iou, compute_iou(pred_mask, gt_mask)) * 1.0
        loss_occ = self.occ_loss(pred_occ, gt_occ) * 1.0
        return {
            'focal': loss_focal,
            'dice': loss_dice, 
            'iou': loss_iou,
            'occ': loss_occ,
            'total': loss_focal + loss_dice + loss_iou + loss_occ,
        }
    
    def _generate_prompt(self, gt_mask, points_xyz):
        r = random.random()
        if r < 0.5:
            return {'type': 'mask', 'data': add_noise_to_mask(gt_mask)}
        elif r < 0.75:
            return {'type': 'point', 'data': sample_points_from_mask(gt_mask, points_xyz)}
        else:
            return {'type': 'box', 'data': mask_to_box_with_jitter(gt_mask, points_xyz)}
        
    def reset_memory(self):
        """每个序列开始时重置memory bank"""
        self.memory_bank = {}

    def _update_memory(self, obj_id, pred_mask, fused_feat):
        if obj_id not in self.memory_bank:
            self.memory_bank[obj_id] = []
        self.memory_bank[obj_id].append({
            'mask': pred_mask,
            'feat': fused_feat
        })
        # 限制memory长度（论文中可能为5-10）
        if len(self.memory_bank[obj_id]) > 10:
            self.memory_bank[obj_id].pop(0)





