import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalLoss(nn.Module):
    """Sigmoid Focal Loss for mask prediction
    论文参数: weight = 20.0, alpha = 0.25, gamma = 2.0 (SAM2默认)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # 论文: SAM2默认值
        self.gamma = gamma  # 论文: SAM2默认值

    def forward(self, pred, target):
        """
        pred: (N,) logits (未sigmoid)
        target: (N,) binary {0, 1}
        """
        p = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * focal_weight * ce_loss
        return loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for mask prediction
    论文参数: weight = 1.0
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: (N,) logits
        target: (N,) binary
        """
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice
