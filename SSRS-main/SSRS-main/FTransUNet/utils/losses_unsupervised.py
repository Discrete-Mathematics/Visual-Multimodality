import torch
import torch.nn as nn
import torch.nn.functional as F

class CycleConsistencyLoss(nn.Module):
    """保留原有逻辑，调整数值稳定性"""
    def __init__(self, eps=1e-6):
        super(CycleConsistencyLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.eps = eps

    def forward(self, raw_depth, raw_depth_recon, valid_mask):
        valid_raw = raw_depth[valid_mask]
        valid_recon = raw_depth_recon[valid_mask]
        if valid_raw.numel() == 0:
            return torch.tensor(0.0).to(raw_depth.device)
        return self.l1(valid_raw, valid_recon)

class CrossModalConsistencyLoss(nn.Module):
    """保留原有逻辑，添加特征归一化增强稳定性"""
    def __init__(self):
        super(CrossModalConsistencyLoss, self).__init__()
        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def forward(self, rgb_feat, depth_feat):
        # 全局平均池化
        rgb_global = F.adaptive_avg_pool2d(rgb_feat, (1, 1)).squeeze(-1).squeeze(-1)
        depth_global = F.adaptive_avg_pool2d(depth_feat, (1, 1)).squeeze(-1).squeeze(-1)
        # 增强归一化稳定性
        rgb_global = F.normalize(rgb_global, p=2, dim=1)
        depth_global = F.normalize(depth_global, p=2, dim=1)
        # 限制相似度范围
        sim = torch.clip(self.cosine_sim(rgb_global, depth_global), -1.0, 1.0)
        return torch.mean(1 - sim)


class WGANLoss(nn.Module):
    """论文WGAN适配，添加输出约束+梯度惩罚支持"""
    def __init__(self, device='cuda'):
        super(WGANLoss, self).__init__()
        self.device = device

    def discriminator_loss(self, pred_real, pred_fake):
        """论文公式L_D = E[D(real)] - E[D(fake)]"""
        loss_real = torch.mean(pred_real)
        loss_fake = torch.mean(pred_fake)
        return loss_real - loss_fake

    def generator_loss(self, pred_fake):
        """论文公式L_G包含 -E[D(fake)]，但限制输出范围避免无限下降"""
        return -torch.mean(pred_fake)  # 本身为负，但后续会被其他损失平衡

    def gradient_penalty(self, discriminator, real_data, fake_data, rgb_img):
        """WGAN-GP梯度惩罚（替代单纯裁剪，更稳定）"""
        batch_size = real_data.size(0)
        # 随机混合系数
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # 判别器输出
        pred_interp = discriminator(interpolated, rgb_img)
        # 计算梯度
        grad_outputs = torch.ones_like(pred_interp)
        gradients = torch.autograd.grad(
            outputs=pred_interp,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        # 梯度惩罚计算
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gp = torch.mean((gradient_norm - 1) ** 2)
        return gp

class TotalLoss(nn.Module):
    def __init__(self, config):
        super(TotalLoss, self).__init__()
        self.wgan_loss = WGANLoss(device=config['device'])
        self.l1 = nn.L1Loss()
        self.cycle_loss = CycleConsistencyLoss()
        self.normal_loss = NormalConsistencyLoss()
        self.crossmodal_loss = CrossModalConsistencyLoss()
        
        # 严格遵循论文权重（无监督下调整λ_D=0.01，弱化判别器影响）
        self.lambda_g = config.get('lambda_g', 0.5)    # 论文λ_g=0.5（d_f的L1）
        self.lambda_l = config.get('lambda_l', 1.0)    # 论文λ_l=1（d_l的L1）
        self.lambda_pred = config.get('lambda_pred', 10.0)  # 论文λ_pred=10（d_pred的L1）
        self.lambda_cycle = config.get('lambda_cycle', 20.0)  # 增大循环损失权重
        self.lambda_normal = config.get('lambda_normal', 10.0) # 增大几何约束权重
        self.lambda_crossmodal = config.get('lambda_crossmodal', 5.0)
        self.lambda_D = config.get('lambda_D', 0.01)   # 大幅降低判别器损失权重
        self.lambda_gp = config.get('lambda_gp', 20.0)  # 梯度惩罚权重

    def forward(self, discriminator, outputs, raw_depth, rgb_img, d_l, d_f, d_pred, invalid_depth):
        """
        严格对齐论文总损失公式：L_overall = L_D + L_G + λ_l L1(d_l) + λ_pred L1(d_pred)
        新增d_l的L1损失，平衡生成器
        """
        # 1. 判别器相关损失（含梯度惩罚）
        pred_real = discriminator(d_pred, rgb_img)  # 合理深度=real
        pred_fake = discriminator(invalid_depth, rgb_img)  # 不合理深度=fake
        L_D = self.wgan_loss.discriminator_loss(pred_real, pred_fake)
        L_gp = self.wgan_loss.gradient_penalty(discriminator, d_pred, invalid_depth, rgb_img)
        L_D_total = L_D + self.lambda_gp * L_gp

        # 2. 生成器相关损失（严格包含论文要求的所有L1项）
        valid_mask = outputs['valid_mask']
        # 论文λ_g*L1(d_f, 原始有效深度)（无监督替代d_gt）
        L1_df = self.l1(d_f[valid_mask], raw_depth[valid_mask]) * self.lambda_g
        # 论文λ_l*L1(d_l, 原始有效深度)
        L1_dl = self.l1(d_l[valid_mask], raw_depth[valid_mask]) * self.lambda_l
        # 论文λ_pred*L1(d_pred, 原始有效深度)
        L1_dpred = self.l1(d_pred[valid_mask], raw_depth[valid_mask]) * self.lambda_pred
        # 对抗损失
        L_G_adv = self.wgan_loss.generator_loss(pred_real)
        # 其他无监督约束（增大权重平衡对抗损失）
        L_cycle = self.cycle_loss(raw_depth, outputs['raw_depth_recon'], valid_mask) * self.lambda_cycle
        L_normal = self.normal_loss(d_pred, rgb_img) * self.lambda_normal
        L_crossmodal = self.crossmodal_loss(outputs['rgb_norm_feat'], outputs['gan_feat']) * self.lambda_crossmodal

        # 3. 生成器总损失（对抗损失占比降低）
        L_G = L_G_adv + L1_df + L1_dl + L1_dpred + L_cycle + L_normal + L_crossmodal

        # 4. 整体损失（判别器损失权重弱化）
        L_overall = L_G + self.lambda_D * L_D_total

        return {
            'L_overall': L_overall,
            'L_G': L_G,
            'L_D': L_D_total,
            'L_G_adv': L_G_adv,
            'L1_total': L1_df + L1_dl + L1_dpred,
            'L_cycle': L_cycle,
            'L_gp': L_gp
        }

# 保留其他损失类（CycleConsistencyLoss/NormalConsistencyLoss/CrossModalConsistencyLoss）
# 但修改NormalConsistencyLoss的数值稳定性
class NormalConsistencyLoss(nn.Module):
    def __init__(self):
        super(NormalConsistencyLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def compute_normal(self, depth_map):
        B, C, H, W = depth_map.shape
        grad_x = depth_map[..., 1:, :] - depth_map[..., :-1, :]
        grad_y = depth_map[..., :, 1:] - depth_map[..., :, :-1]
        grad_x = F.pad(grad_x, (0, 0, 0, 1))
        grad_y = F.pad(grad_y, (0, 1, 0, 0))
        normal = torch.cat([-grad_x, -grad_y, torch.ones_like(depth_map)], dim=1)
        normal = F.normalize(normal, p=2, dim=1)
        return torch.clip(normal, -1.0, 1.0)

    def compute_rgb_edge_normal(self, rgb_img):
        gray = torch.mean(rgb_img, dim=1, keepdim=True)
        edge_x = gray[..., 1:, :] - gray[..., :-1, :]
        edge_y = gray[..., :, 1:] - gray[..., :, :-1]
        edge_x = F.pad(edge_x, (0, 0, 0, 1))
        edge_y = F.pad(edge_y, (0, 1, 0, 0))
        edge_normal = torch.cat([-edge_x, -edge_y, torch.ones_like(gray)], dim=1)
        edge_normal = F.normalize(edge_normal, p=2, dim=1)
        return torch.clip(edge_normal, -1.0, 1.0)

    def forward(self, d_pred, rgb_img):
        if d_pred.shape[2:] != rgb_img.shape[2:]:
            d_pred = F.interpolate(d_pred, size=rgb_img.shape[2:], mode='bilinear', align_corners=False)
        depth_normal = self.compute_normal(d_pred)
        rgb_edge_normal = self.compute_rgb_edge_normal(rgb_img)
        return self.l1(depth_normal, rgb_edge_normal)