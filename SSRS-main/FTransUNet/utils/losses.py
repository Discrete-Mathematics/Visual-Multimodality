import torch
import torch.nn as nn
import torch.nn.functional as F

class WGANLoss(nn.Module):
    """论文3.2节指定：RDF-GAN采用WGAN目标函数"""
    def __init__(self, device='cuda'):
        super(WGANLoss, self).__init__()
        self.device = device

    def discriminator_loss(self, pred_real, pred_fake):
        """论文公式1：L_D = E[D(G(M(d_raw))|r)] - E[D(d_gt|r)]"""
        loss_real = torch.mean(pred_real)
        loss_fake = torch.mean(pred_fake)
        return loss_fake - loss_real  # WGAN判别器损失：最大化 E[D(real)] - E[D(fake)]

    def generator_loss(self, pred_fake):
        """论文公式2：L_G 包含对抗损失部分（-E[D(G(M(d_raw))|r)]）"""
        return -torch.mean(pred_fake)

class DepthL1Loss(nn.Module):
    """论文3.6节：L1损失（局部深度图d_l + 最终预测d_pred）"""
    def __init__(self):
        super(DepthL1Loss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.l1(pred, target)



# 深度补全L1损失
class DepthL1Loss(nn.Module):
    def __init__(self):
        super(DepthL1Loss, self).__init__()
        self.l1 = nn.L1Loss()
    
    def forward(self, fake_depth, real_depth):
        return self.l1(fake_depth, real_depth)

# 语义分割交叉熵损失
class SegCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=19):
        super(SegCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, pred_seg, seg_gt):
        # pred_seg: [B, C, H, W], seg_gt: [B, H, W]
        return F.cross_entropy(pred_seg, seg_gt, ignore_index=-1)

# GAN对抗损失（LSGAN）
class GANLoss(nn.Module):
    def __init__(self, gan_mode='lsgan', device='cuda'):
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        self.device = device
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"GAN mode {gan_mode} not supported")
    
    def forward(self, pred, target_real):
        target_tensor = torch.tensor(1.0).expand_as(pred).to(self.device) if target_real else torch.tensor(0.0).expand_as(pred).to(self.device)
        return self.loss(pred, target_tensor)