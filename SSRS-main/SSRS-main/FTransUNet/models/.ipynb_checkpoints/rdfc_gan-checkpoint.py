import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
from utils.losses_unsupervised import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """论文核心残差块（适配RGB图像增强）：保持特征维度，提升梯度传播能力"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect')
        self.bn1 = nn.InstanceNorm2d(channels)  # 实例归一化：适配图像增强任务，避免BatchNorm的批次依赖
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect')
        self.bn2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual  # 残差连接：避免深层网络退化
        x = self.relu(x)
        return x

class GeneratorB2A(nn.Module):
    """
    GeneratorB2A：RGB图像增强生成器（B→A：原始RGB → 增强RGB）
    贴合论文设计：ResNet-18风格编码器-解码器 + 残差块 + 跳跃连接
    输入/输出：[B, 3, H, W]（适配256×256尺寸，与你的数据集对齐）
    """
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_res_blocks=9):
        super(GeneratorB2A, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngf = ngf  # 初始特征通道数（论文默认64）

        # ---------------------- 编码器（下采样：4次，特征图尺寸→H/16，通道数→ngf×8） ----------------------
        # 输入层：反射填充+Conv+InstanceNorm+ReLU（避免边缘失真）
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=3, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        # 下采样1：ngf → ngf×2，尺寸/2
        self.conv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(inplace=True)
        )
        # 下采样2：ngf×2 → ngf×4，尺寸/2
        self.conv3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(inplace=True)
        )
        # 下采样3：ngf×4 → ngf×8，尺寸/2
        self.conv4 = nn.Sequential(
            nn.Conv2d(ngf*4, ngf*8, kernel_size=3, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(ngf*8),
            nn.ReLU(inplace=True)
        )
        # 下采样4：ngf×8 → ngf×8，尺寸/2（最终特征图尺寸：H/16 × W/16）
        self.conv5 = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(ngf*8),
            nn.ReLU(inplace=True)
        )

        # ---------------------- 中间特征提取（残差块：9个，论文默认配置，提升特征表达能力） ----------------------
        self.res_blocks = nn.Sequential(*[ResidualBlock(ngf*8) for _ in range(n_res_blocks)])

        # ---------------------- 解码器（上采样：4次，特征图尺寸→H，通道数→out_channels） ----------------------
        # 上采样1：ngf×8 → ngf×8，尺寸×2
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf*8),
            nn.ReLU(inplace=True)
        )
        # 上采样2：ngf×8 → ngf×4，尺寸×2
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(inplace=True)
        )
        # 上采样3：ngf×4 → ngf×2，尺寸×2
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(inplace=True)
        )
        # 上采样4：ngf×2 → ngf，尺寸×2
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )

        # ---------------------- 输出层（生成增强RGB图像） ----------------------
        # Tanh激活：将输出归一化到[-1, 1]，后续可映射回[0, 1]适配RGB格式
        self.out_conv = nn.Sequential(
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=3, bias=False, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x):
        """
        前向传播：带跳跃连接，保留编码器细节信息，提升增强图像的纹理完整性
        x: 原始RGB图像 [B, 3, H, W]
        """
        # 编码器前向（保存跳跃连接特征）
        feat1 = self.conv1(x)          # [B, ngf, H, W]
        feat2 = self.conv2(feat1)      # [B, ngf×2, H/2, W/2]
        feat3 = self.conv3(feat2)      # [B, ngf×4, H/4, W/4]
        feat4 = self.conv4(feat3)      # [B, ngf×8, H/8, W/8]
        feat5 = self.conv5(feat4)      # [B, ngf×8, H/16, W/16]

        # 中间残差特征提取
        res_feat = self.res_blocks(feat5)  # [B, ngf×8, H/16, W/16]

        # 解码器前向（跳跃连接：拼接编码器对应层特征，保留细节）
        dec_feat1 = self.deconv1(res_feat) + feat4  # [B, ngf×8, H/8, W/8]
        dec_feat2 = self.deconv2(dec_feat1) + feat3  # [B, ngf×4, H/4, W/4]
        dec_feat3 = self.deconv3(dec_feat2) + feat2  # [B, ngf×2, H/2, W/2]
        dec_feat4 = self.deconv4(dec_feat3) + feat1  # [B, ngf, H, W]

        # 生成最终增强RGB图像
        enhanced_rgb = self.out_conv(dec_feat4)  # [B, 3, H, W]
        return enhanced_rgb

# ===================== 论文核心模块：局部引导模块（3.3节，无监督适配） =====================
class LocalGuidanceModule(nn.Module):
    """从RGB提取2通道引导图（前景概率+边缘特征），无监督下引导深度补全"""
    def __init__(self, in_channels=3, out_channels=2, ngf=64):
        super(LocalGuidanceModule, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Conv2d(ngf, out_channels, 7, 1, 3, bias=False)
        
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False),  # 128->256
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, rgb_img):
        e1 = self.enc1(rgb_img)      # 128
        e2 = self.enc2(e1)           # 64
        e3 = self.enc3(e2)           # 32
        d1 = self.dec1(e3)           # 64
        d2 = self.dec2(d1 + e2)      # 128
        g = self.dec3(d2 + e1)       # 128
        g = self.final_upsample(g)   # 256 ← 添加这行
        return g  # [B, 2, 256, 256]

# ===================== 论文核心模块：W-AdaIN（3.3节，无监督适配） =====================
class WAdaIN(nn.Module):
    """无监督下保留跨模态融合逻辑，保证RGB与深度特征匹配"""
    def __init__(self, in_channels_rgb, in_channels_z, out_channels, attn_size=64):
        super(WAdaIN, self).__init__()
        self.out_channels = out_channels
        self.attn_size = attn_size  # 【新增】限制注意力计算的分辨率
        
        # 添加投影层：将不同来源的特征统一到 out_channels
        self.rgb_proj = nn.Conv2d(in_channels_rgb, out_channels, kernel_size=1)
        self.z_proj = nn.Conv2d(in_channels_z, out_channels, kernel_size=1)
        
        self.self_attn_z = SelfAttention(out_channels)      
        self.self_attn_rgb = SelfAttention(out_channels)    
        
        self.affine = nn.Sequential(
            nn.Linear(in_channels_z, out_channels*2),       
            nn.ReLU(inplace=True)
        )

    def forward(self, z, f_r):
        # 1. 将 z 上采样到 f_r 的空间尺寸
        z_up = F.interpolate(z, size=f_r.shape[2:], mode='bilinear', align_corners=False)
        
        # 2. 投影到统一的通道数 out_channels
        z_proj = self.z_proj(z_up)        # [B, out_channels, H, W]
        f_proj = self.rgb_proj(f_r)       # [B, out_channels, H, W]
        
        # 3. 【关键修复】在低分辨率下计算自注意力，避免OOM
        if z_proj.shape[2] > self.attn_size:
            # 下采样到指定分辨率（如64x64）
            z_for_attn = F.adaptive_avg_pool2d(z_proj, (self.attn_size, self.attn_size))
            f_for_attn = F.adaptive_avg_pool2d(f_proj, (self.attn_size, self.attn_size))
            
            # 计算注意力
            A_small = self.self_attn_z(z_for_attn)
            B_small = self.self_attn_rgb(f_for_attn)
            
            # 上采样回原始尺寸
            A = F.interpolate(A_small, size=z_proj.shape[2:], mode='bilinear', align_corners=False)
            B = F.interpolate(B_small, size=f_proj.shape[2:], mode='bilinear', align_corners=False)
        else:
            # 如果本身分辨率就不大，直接计算
            A = self.self_attn_z(z_proj)
            B = self.self_attn_rgb(f_proj)
        
        # 4. 对投影后的 RGB 特征进行归一化
        mu = f_proj.mean(dim=[2, 3], keepdim=True)
        sigma = f_proj.std(dim=[2, 3], keepdim=True) + 1e-6
        f_r_norm = (f_proj - mu) / sigma
        
        # 5. 从原始 z 提取全局统计量，生成调制参数
        z_global = z.mean(dim=[2, 3])
        y = self.affine(z_global)
        y_s = y[:, :self.out_channels].unsqueeze(2).unsqueeze(3)
        y_b = y[:, self.out_channels:].unsqueeze(2).unsqueeze(3)
        
        # 6. 融合
        out = torch.mul(A, torch.mul(y_s, f_r_norm)) + torch.mul(B, y_b)
        return out, f_r_norm

# Helper函数：创建带downsample的BasicBlock
def make_basic_block(in_ch, out_ch, stride=1):
    downsample = None
    if stride != 1 or in_ch != out_ch:
        downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            )
    return BasicBlock(in_ch, out_ch, stride=stride, downsample=downsample)

# ===================== 论文分支1：约束网络（3.1节，无监督适配） =====================
class ConstraintNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels_depth=1, out_channels_conf=1, ngf=64):
        super(ConstraintNetwork, self).__init__()
        
        # Encoder（ResNet-18前4层）- 使用make_basic_block替换原来的BasicBlock
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, ngf, 7, 2, 3, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            make_basic_block(ngf, ngf, stride=1),
            make_basic_block(ngf, ngf, stride=1),
            make_basic_block(ngf, ngf*2, stride=2),  # 下采样，需要downsample
            make_basic_block(ngf*2, ngf*2, stride=1),
            make_basic_block(ngf*2, ngf*4, stride=2), # 下采样，需要downsample
            make_basic_block(ngf*4, ngf*4, stride=1),
            make_basic_block(ngf*4, ngf*8, stride=2), # 下采样，需要downsample
            make_basic_block(ngf*8, ngf*8, stride=1),
        )
        # ... 后续代码保持不变
        # Decoder（补全深度+置信图）
        self.decoder_depth = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, out_channels_depth, 7, 1, 3, bias=False)
        )
        # 置信图输出
        # 在 ConstraintNetwork.__init__ 中修改 decoder_conf
        self.decoder_conf = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),  # 8->16
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),  # 16->32
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),    # 32->64
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),      # 64->128
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),      # 128->256
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, out_channels_conf, 1, 1, 0, bias=False),  # 1x1 卷积输出置信图
            nn.Sigmoid()
        )
        # 新增：循环生成分支（从d_l逆生成残缺深度raw_depth_recon）
        # 在 ConstraintNetwork.__init__ 中修改 cycle_decoder：
        self.cycle_decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),  # 8->16
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),  # 16->32
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),    # 32->64
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),      # 64->128 【新增】
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, out_channels_depth, 4, 2, 1, bias=False),  # 128->256
            nn.Tanh()
        )

    def forward(self, x):
        feat = self.encoder(x)
        d_l = self.decoder_depth(feat)
        c_l = self.decoder_conf(feat)
        raw_depth_recon = self.cycle_decoder(feat)  # 循环生成残缺深度
        #print(x.shape, feat.shape, d_l.shape, c_l.shape)
        return d_l, c_l, raw_depth_recon, feat  # 返回feat用于跨模态损失

# ===================== 论文分支2：RDF-GAN生成器（3.2节，无监督适配） =====================
class RDFGANGenerator(nn.Module):
    """无监督下：生成融合深度d_f和置信图c_f，输出融合特征用于跨模态损失"""
    def __init__(self, in_channels_z=512, in_channels_rgb=3, ngf=64):
        super(RDFGANGenerator, self).__init__()
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(in_channels_rgb, ngf, 7, 1, 3, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        # 投影层
        self.proj_fused2 = nn.Conv2d(ngf*4, ngf*2, kernel_size=1)  # 256 -> 128
        # W-AdaIN融合模块
        self.wadain1 = WAdaIN(in_channels_rgb=ngf, in_channels_z=in_channels_z, out_channels=ngf*2)
        self.wadain2 = WAdaIN(in_channels_rgb=ngf*2, in_channels_z=in_channels_z, out_channels=ngf*4)
        # 主干网络
        # 在 RDFGANGenerator.__init__ 中修改 backbone
        self.backbone = nn.Sequential(
            make_basic_block(ngf*2, ngf*2, stride=1),      # 256x256
            make_basic_block(ngf*2, ngf*4, stride=2),      # 128x128
            make_basic_block(ngf*4, ngf*4, stride=1),      # 128x128
            make_basic_block(ngf*4, ngf*8, stride=2),      # 64x64
            make_basic_block(ngf*8, ngf*8, stride=1),      # 64x64
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),  # 64->128
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),  # 128->256
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            # 去掉后两个 ConvTranspose2d（避免 128->256->512），改为普通卷积保持 256x256
            nn.Conv2d(ngf*2, ngf, 3, 1, 1, bias=False),    # 256x256, 256->64 channels
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),  # 64->128
            # nn.BatchNorm2d(ngf*2),
            # nn.ReLU(inplace=True),
        )
        # 输出头
        self.out_depth = nn.Conv2d(ngf, 1, 7, 1, 3, bias=False)
        self.out_conf = nn.Conv2d(ngf, 1, 7, 1, 3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, rgb_img):
        f_r = self.rgb_encoder(rgb_img)
        f_fused1, f_r_norm1 = self.wadain1(z, f_r)
        f_fused2, f_r_norm2 = self.wadain2(z, f_fused1)
        
        # 【关键修复】保存投影前的 256 通道特征用于跨模态损失计算
        gan_feat_for_loss = f_fused2  # 通道数为 256
        
        # 投影回 128 通道以匹配 backbone 输入要求
        f_fused2_proj = self.proj_fused2(f_fused2) 
        feat = self.backbone(f_fused2_proj)
        
        d_f = self.out_depth(feat)
        c_f = self.sigmoid(self.out_conf(feat))
        
        # 返回 256 通道的 gan_feat_for_loss 和 f_r_norm2
        return d_f, c_f, gan_feat_for_loss, f_r_norm2

# ===================== 论文分支2：RDF-GAN判别器（3.2节，无监督适配） =====================
class RDFGANDiscriminator(nn.Module):
    """无监督判别器：区分「合理补全深度」与「不合理深度」（随机噪声/无效深度）"""
    def __init__(self, in_channels_depth=1, in_channels_rgb=3, ndf=64, n_layers=3):
        super(RDFGANDiscriminator, self).__init__()
        self.in_channels = in_channels_depth + in_channels_rgb  # 深度+RGB拼接输入
        layers = []
        # 第一层无BN
        layers.append(nn.Conv2d(self.in_channels, ndf, 4, 2, 1, bias=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # 后续层带BN
        mult = 1
        for i in range(1, n_layers):
            prev_mult = mult
            mult = min(2**i, 8)
            layers.extend([
                nn.Conv2d(ndf*prev_mult, ndf*mult, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf*mult),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        # 输出Patch级判别结果
        layers.append(nn.Conv2d(ndf*mult, 1, 4, 1, 1, bias=True))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, depth_img, rgb_img):
        """输入：深度图（合理/不合理）+ RGB图，输出Patch级判别分数"""
        x = torch.cat([depth_img, rgb_img], dim=1)
        return self.model(x)

# ===================== 论文核心：置信融合头（3.4节，无变化） =====================
class ConfidenceFusionHead(nn.Module):
    def __init__(self, eps=1e-8):
        super(ConfidenceFusionHead, self).__init__()
        self.eps = eps

    def forward(self, d_l, c_l, d_f, c_f):
        weight_l = torch.exp(c_l)
        weight_f = torch.exp(c_f)
        #print([x.shape for x in [weight_l, d_l, weight_f, d_f]])
        numerator = weight_l * d_l + weight_f * d_f
        denominator = weight_l + weight_f + self.eps
        d_pred = numerator / denominator
        return d_pred

# ===================== 辅助模块：自注意力（W-AdaIN依赖） =====================
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.query_conv = nn.Conv2d(channels, channels//8, 1)
        self.key_conv = nn.Conv2d(channels, channels//8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        query = self.query_conv(x).view(B, -1, H*W).permute(0, 2, 1)
        key = self.key_conv(x).view(B, -1, H*W)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        value = self.value_conv(x).view(B, -1, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

# ===================== 完整无监督RDFC-GAN模型 =====================
class UnsupervisedRDFCGAN(nn.Module):
    def __init__(self, config):
        super(UnsupervisedRDFCGAN, self).__init__()
        self.config = config
        # 论文核心模块
        self.local_guidance = LocalGuidanceModule()
        self.constraint_net = ConstraintNetwork(in_channels=3)
        self.rdfgan_gen = RDFGANGenerator(in_channels_z=512)
        self.rdfgan_disc = RDFGANDiscriminator()
        self.conf_fusion = ConfidenceFusionHead()
        # 无监督损失函数
        self.wgan_loss = WGANLoss(device=config['device'])
        self.cycle_loss = CycleConsistencyLoss()
        self.normal_loss = NormalConsistencyLoss()
        self.crossmodal_loss = CrossModalConsistencyLoss()
        self.total_loss = TotalLoss(config)

    def get_depth_latent(self, x):
        """提取深度latent向量z（论文3.1节M(·)）"""
        for i, layer in enumerate(self.constraint_net.encoder[:12]):
            x = layer(x)
        return x

    def generate_invalid_depth(self, d_pred):
        """生成不合理深度（用于判别器训练）：随机噪声+局部扰动"""
        B, C, H, W = d_pred.shape
        noise = torch.randn_like(d_pred) * 0.1
        invalid_depth = d_pred + noise
        # 随机遮挡部分区域，模拟无效深度
        mask = torch.rand(B, 1, H, W).to(d_pred.device) > 0.7
        invalid_depth[mask] = torch.randn_like(invalid_depth[mask])
        return invalid_depth

    def forward(self, raw_depth, rgb_img):
        """前向传播：raw_depth（残缺深度）+ rgb_img → d_pred（补全深度）"""
        # 1. 局部引导图
        g = self.local_guidance(rgb_img)
        # 2. 约束网络输入：raw_depth + g
        constraint_input = torch.cat([raw_depth, g], dim=1)
        d_l, c_l, raw_depth_recon, constraint_feat = self.constraint_net(constraint_input)
        
        # 3. 提取latent向量z
        z = self.get_depth_latent(constraint_input)
        # 4. RDF-GAN生成器
        #print(z.shape, rgb_img.shape)
        d_f, c_f, gan_feat, rgb_norm_feat = self.rdfgan_gen(z, rgb_img)
        # 5. 置信融合
        d_pred = self.conf_fusion(d_l, c_l, d_f, c_f)
        # 6. 有效像素掩码（基于原始 raw_depth）
        valid_mask = (raw_depth != 0) & (torch.isfinite(raw_depth))
        
        return {
            'd_pred': d_pred, 'd_l': d_l, 'd_f': d_f,
            'raw_depth_recon': raw_depth_recon,  # 现在尺寸与 raw_depth 匹配：[B,1,256,256]
            'valid_mask': valid_mask,
            'constraint_feat': constraint_feat, 
            'gan_feat': gan_feat,
            'rgb_norm_feat': rgb_norm_feat
        }
    
    def generate_invalid_depth(self, d_pred):
        """优化不合理深度生成，确保判别器能区分"""
        B, C, H, W = d_pred.shape
        # 1. 随机噪声扰动（增强不合理性）
        noise = torch.randn_like(d_pred) * 0.3  # 适当增大噪声
        invalid_depth = d_pred + noise
        # 2. 随机大块遮挡（模拟传感器严重缺失）
        mask = torch.rand(B, 1, H, W).to(d_pred.device) > 0.5  # 遮挡比例提升到50%
        invalid_depth[mask] = torch.randn_like(invalid_depth[mask]) * 0.5
        # 3. 深度值裁剪（避免极端值）
        invalid_depth = torch.clip(invalid_depth, -1.0, 2.0)
        return invalid_depth

    def calculate_loss(self, raw_depth, rgb_img):
        """修正损失计算流程，添加梯度裁剪"""
        outputs = self.forward(raw_depth, rgb_img)
        d_pred = outputs['d_pred']
        invalid_depth = self.generate_invalid_depth(d_pred)

        # 计算总损失
        loss_dict = self.total_loss(
            discriminator=self.rdfgan_disc,
            outputs=outputs,
            raw_depth=raw_depth,
            rgb_img=rgb_img,
            d_l=outputs["d_l"],
            d_f=outputs["d_f"],
            d_pred=d_pred,
            invalid_depth=invalid_depth
        )

        # 保存d_pred用于验证
        loss_dict['d_pred'] = d_pred
        return loss_dict
        