import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
# 导入配置和模块
from config import *
from models.backbone import ResNet50
from models.fusion import CBAM
from models.segmentation import DeepLabV3Plus_Head
from utils.init import init_weights
from utils.losses import DepthL1Loss, SegCrossEntropyLoss, GANLoss
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2          
from PIL import Image, ImageFilter
from utils.metrics import (
    calculate_ssim, calculate_gradient_similarity, calculate_phash_similarity,
    calculate_pixel_accuracy, calculate_miou
)
from models.rdfc_gan import UnsupervisedRDFCGAN, UNSUPERVISED_CONFIG

# 初始化模型
depth_model = UnsupervisedRDFCGAN, UNSUPERVISED_CONFIG(UNSUPERVISED_CONFIG).to(DEVICE)

# 定义优化器（生成器：约束网络+RDF-GAN生成器；判别器：仅RDF-GAN判别器）
optimizer_gen = torch.optim.Adam(
    list(model.constraint_net.parameters()) + list(model.rdfgan_gen.parameters()),
    lr=1e-4, betas=(0.5, 0.999)
)
optimizer_disc = torch.optim.Adam(
    model.rdfgan_disc.parameters(),
    lr=1e-4, betas=(0.5, 0.999)
)

STAGE1_EPOCHS = 1
STAGE2_EPOCHS = 1
STAGE3_EPOCHS = 1

def read_txt_to_list(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f'读取文件时出错: {e}')
        return []
        
class EndToEndDataset(Dataset):
    def __init__(self, ids, rgb_folder, depth_folder, label_folder,
                 target_size=(256, 256), max_depth_val=65535):  # 添加max_depth_val参数
        self.ids = ids
        self.rgb_fmt = rgb_folder
        self.depth_fmt = depth_folder
        self.label_fmt = label_folder
        self.target_size = target_size
        self.max_depth_val = max_depth_val

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        # RGB处理
        rgb = Image.open(self.rgb_fmt.format(id_)).convert('RGB')
        rgb = rgb.resize(self.target_size, Image.BICUBIC)
        rgb_tensor = self.rgb_transform(rgb)

        # 深度图处理（16-bit）
        depth16 = cv2.imread(self.depth_fmt.format(id_), cv2.IMREAD_ANYDEPTH)
        if depth16 is None:
            raise FileNotFoundError(f"无法读取深度图: {self.depth_fmt.format(id_)}")
        
        depth16 = cv2.resize(depth16, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # 创建缺失区域掩码（0=缺失，1=有效）
        mask_missing = (depth16 == 0).astype(np.float32)
        
        # 归一化到[0,1]
        depth_arr = depth16.astype(np.float32) / self.max_depth_val
        depth_arr = np.clip(depth_arr, 0.0, 1.0)

        depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0)  # [1, H, W]
        mask_missing_tensor = torch.from_numpy(mask_missing).unsqueeze(0)  # [1, H, W]

        # 语义标签
        label = Image.open(self.label_fmt.format(id_)).convert('L')
        label = label.resize(self.target_size, Image.NEAREST)
        label_tensor = torch.from_numpy(np.array(label, dtype=np.uint8)).long()

        return rgb_tensor, depth_tensor, label_tensor, mask_missing_tensor        

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ---- 数据路径 ----
# 使用您现有的数据集格式
DATA_ROOT = '/root/autodl-tmp/dataset'
rgb_fmt = os.path.join(DATA_ROOT, 'Image', '{}.jpg')
depth_fmt = os.path.join(DATA_ROOT, 'Depth', '{}.png')
label_fmt = os.path.join(DATA_ROOT, 'Label_Label', '{}_label.png')

# 读取数据ID
train_ids = read_txt_to_list('train.txt')
val_ids = read_txt_to_list('val.txt')
test_ids = read_txt_to_list('test.txt')
# 根据你的数据集调整这个值
MAX_DEPTH_VAL = 65535  # 对于16-bit深度图通常是65535或10000
    
train_set = EndToEndDataset(
    train_ids, rgb_fmt, depth_fmt, label_fmt, 
    target_size=(256, 256),
    max_depth_val=MAX_DEPTH_VAL  # 显式指定
)
val_set = EndToEndDataset(val_ids, rgb_fmt, depth_fmt, label_fmt, target_size=(256, 256))
test_set = EndToEndDataset(test_ids, rgb_fmt, depth_fmt, label_fmt, target_size=(256, 256))

BATCH_SIZE = 2
NUM_WORKERS = 4

train_dataloader = iter(DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True))
val_dataloader = iter(DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True))
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
def validate_stage1(netG_A2B):
    """
    阶段1验证：仅无监督深度补全指标
    """
    print("----- 开始阶段1验证（无监督深度补全）-----")
    netG_A2B.eval()  # 切换为评估模式，关闭BN/Dropout
    ssim_scores = []
    grad_scores = []
    phash_scores = []
    
    with torch.no_grad():  # 关闭梯度计算，节省显存和计算资源
        # 遍历验证集（取固定批次，避免无限循环）
        for _ in range(50):  # 50批次足够反映验证集整体效果
            _, raw_depth, _ = next(val_dataloader)
            fake_depth = netG_A2B(raw_depth)
            
            # 计算无监督指标
            ssim_score = calculate_ssim(raw_depth, fake_depth)
            grad_score = calculate_gradient_similarity(raw_depth, fake_depth)
            phash_score = calculate_phash_similarity(raw_depth, fake_depth)
            
            ssim_scores.append(ssim_score)
            grad_scores.append(grad_score)
            phash_scores.append(phash_score)
    
    # 计算平均指标
    avg_ssim = np.mean(ssim_scores)
    avg_grad = np.mean(grad_scores)
    avg_phash = np.mean(phash_scores)
    
    print(f"阶段1验证结果 | SSIM: {avg_ssim:.4f} | Grad SIM: {avg_grad:.4f} | pHash SIM: {avg_phash:.4f}")
    netG_A2B.train()  # 切换回训练模式
    return avg_ssim  # 以SSIM作为核心最优指标（可替换为其他）
    
def validate_stage2(netG_A2B, backbone_rgb, backbone_depth, fusion_module, seg_head):
    """
    阶段2验证：无监督深度补全 + 有监督语义分割
    """
    print("----- 开始阶段2验证（深度+分割）-----")
    # 切换eval模式
    netG_A2B.eval()
    backbone_rgb.eval()
    backbone_depth.eval()
    fusion_module.eval()
    seg_head.eval()
    
    # 初始化指标列表
    ssim_scores = []
    grad_scores = []
    pixel_acc_scores = []
    miou_scores = []
    compress_conv = nn.Conv2d(4096, 2048, kernel_size=1, bias=False).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(50):
            rgb_img, raw_depth, seg_gt = next(val_dataloader)
            fake_depth = netG_A2B(raw_depth)
            
            # 深度补全指标计算
            ssim_score = calculate_ssim(raw_depth, fake_depth)
            grad_score = calculate_gradient_similarity(raw_depth, fake_depth)
            
            # 语义分割前向传播
            feat_rgb = backbone_rgb(rgb_img)
            feat_depth = backbone_depth(fake_depth)
            fused_input = torch.cat([feat_rgb, feat_depth], dim=1)
            fused_feat = compress_conv(fused_input)
            refined_feat = fusion_module(fused_feat)
            pred_seg = seg_head(refined_feat)
            
            # 分割指标计算
            pixel_acc = calculate_pixel_accuracy(pred_seg, seg_gt, ignore_index=-1)
            miou = calculate_miou(pred_seg, seg_gt, num_classes=NUM_CLASSES, ignore_index=-1)
            
            # 累计指标
            ssim_scores.append(ssim_score)
            grad_scores.append(grad_score)
            pixel_acc_scores.append(pixel_acc)
            miou_scores.append(miou)
    
    # 平均指标
    avg_ssim = np.mean(ssim_scores)
    avg_grad = np.mean(grad_scores)
    avg_pixel_acc = np.mean(pixel_acc_scores)
    avg_miou = np.mean(miou_scores)
    
    print(f"阶段2验证结果 | SSIM: {avg_ssim:.4f} | Grad SIM: {avg_grad:.4f} | Pixel Acc: {avg_pixel_acc:.4f} | mIoU: {avg_miou:.4f}")
    # 切换回训练模式
    netG_A2B.train()
    backbone_rgb.train()
    backbone_depth.train()
    fusion_module.train()
    seg_head.train()
    
    return avg_miou  # 以mIoU作为核心最优指标（分割任务核心）

def validate_stage3(netG_A2B, netG_B2A, backbone_rgb, backbone_depth, fusion_module, seg_head):
    """
    阶段3验证：全模块（深度+图像增强+分割）
    """
    print("----- 开始阶段3验证（全模块）-----")
    # 切换eval模式
    netG_A2B.eval()
    netG_B2A.eval()
    backbone_rgb.eval()
    backbone_depth.eval()
    fusion_module.eval()
    seg_head.eval()
    
    # 初始化指标列表
    depth_ssim_scores = []
    rgb_ssim_scores = []
    miou_scores = []
    compress_conv = nn.Conv2d(4096, 2048, kernel_size=1, bias=False).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(50):
            rgb_img, raw_depth, seg_gt = next(val_dataloader)
            fake_depth = netG_A2B(raw_depth)
            fake_rgb = netG_B2A(rgb_img)
            
            # 深度补全指标
            depth_ssim = calculate_ssim(raw_depth, fake_depth)
            
            # 图像增强指标（输入RGB vs 增强RGB）
            rgb_ssim = calculate_ssim(rgb_img, fake_rgb)
            
            # 分割前向传播
            feat_rgb = backbone_rgb(fake_rgb)
            feat_depth = backbone_depth(fake_depth)
            fused_input = torch.cat([feat_rgb, feat_depth], dim=1)
            fused_feat = compress_conv(fused_input)
            refined_feat = fusion_module(fused_feat)
            pred_seg = seg_head(refined_feat)
            
            # 分割指标
            miou = calculate_miou(pred_seg, seg_gt, num_classes=NUM_CLASSES, ignore_index=-1)
            
            # 累计指标
            depth_ssim_scores.append(depth_ssim)
            rgb_ssim_scores.append(rgb_ssim)
            miou_scores.append(miou)
    
    # 平均指标
    avg_depth_ssim = np.mean(depth_ssim_scores)
    avg_rgb_ssim = np.mean(rgb_ssim_scores)
    avg_miou = np.mean(miou_scores)
    
    print(f"阶段3验证结果 | Depth SSIM: {avg_depth_ssim:.4f} | RGB SSIM: {avg_rgb_ssim:.4f} | mIoU: {avg_miou:.4f}")
    # 切换回训练模式
    netG_A2B.train()
    netG_B2A.train()
    backbone_rgb.train()
    backbone_depth.train()
    fusion_module.train()
    seg_head.train()
    
    return avg_miou  # 以mIoU作为全模块核心最优指标
    
# ===================== 阶段1：深度补全预热训练 =====================
def train_stage1():
    print("="*60)
    print("开始阶段1：深度补全预热训练")
    print("="*60)

    init_weights(depth_model, init_type='normal', init_gain=0.02)

    # 3. 开始训练
    best_ssim = -1
    
    for epoch in range(1, STAGE1_EPOCHS+1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_l1_loss = 0.0

        # 批量训练（tqdm进度条）
        pbar = tqdm(range(100), desc=f"Stage1 Epoch {epoch}/{STAGE1_EPOCHS}")
        for _ in pbar:
            # 加载数据
            rgb_img, raw_depth, _, _ = next(train_dataloader)
            real_depth = raw_depth  # 模拟真实深度标签（实际需替换为真实深度GT）

            # 训练生成器（约束网络+RDF-GAN生成器）
            optimizer_gen.zero_grad()
            loss_dict, d_pred = depth_model.calculate_loss(raw_depth, rgb_img)
            loss_dict['L_overall'].backward()
            optimizer_gen.step()

            # 对抗损失
            pred_fake = netD_A(fake_depth)
            loss_g_gan = criterion_gan(pred_fake, target_real=True)

            # L1深度损失
            loss_g_l1 = criterion_l1(fake_depth, real_depth) * LAMBDA_L1

            # 总生成器损失
            loss_g_total = loss_g_gan + loss_g_l1
            loss_g_total.backward()
            optimizer_G.step()

            # ----------------------
            # 训练判别器 D_A
            # ----------------------
            optimizer_D.zero_grad()

            # 真实深度
            pred_real = netD_A(real_depth)
            loss_d_real = criterion_gan(pred_real, target_real=True)

            # 虚假深度（冻结生成器）
            fake_depth_detach = fake_depth.detach()
            pred_fake = netD_A(fake_depth_detach)
            loss_d_fake = criterion_gan(pred_fake, target_real=False)

            # 总判别器损失
            loss_d_total = (loss_d_real + loss_d_fake) * 0.5
            loss_d_total.backward()
            optimizer_D.step()

            # ----------------------
            # 累计损失
            # ----------------------
            epoch_g_loss += loss_g_total.item()
            epoch_d_loss += loss_d_total.item()
            epoch_l1_loss += loss_g_l1.item()

            # 更新进度条
            pbar.set_postfix({
                "G Loss": f"{loss_g_total.item():.4f}",
                "D Loss": f"{loss_d_total.item():.4f}",
                "L1 Loss": f"{loss_g_l1.item():.4f}"
            })

        # ----------------------
        # 每个Epoch结束后：打印平均损失、保存权重
        # ----------------------
        avg_g_loss = epoch_g_loss / 100
        avg_d_loss = epoch_d_loss / 100
        avg_l1_loss = epoch_l1_loss / 100
        print(f"Stage1 Epoch {epoch} | Avg G Loss: {avg_g_loss:.4f} | Avg D Loss: {avg_d_loss:.4f} | Avg L1 Loss: {avg_l1_loss:.4f}")

        # 验证
        if epoch % VAL_EVERY_N_EPOCHS == 0:
            current_ssim = validate_stage1(netG_A2B)
            
            # 6. 新增：保存最优模型
            if current_ssim > best_ssim:
                best_ssim = current_ssim
                torch.save({
                    "epoch": epoch,
                    "netG_A2B": netG_A2B.state_dict(),
                    "netD_A": netD_A.state_dict(),
                    "best_ssim": best_ssim,
                    "optimizer_G": optimizer_G.state_dict()
                }, STAGE1_CKPT_PATH)
                print(f"阶段1最优模型更新！当前最优SSIM: {best_ssim:.4f}，已保存至 {STAGE1_CKPT_PATH}")
                
        # 保存最终权重
        if epoch == STAGE1_EPOCHS:
            torch.save({
                "epoch": epoch,
                "netG_A2B": netG_A2B.state_dict(),
                "netD_A": netD_A.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
                "avg_l1_loss": avg_l1_loss
            }, STAGE1_CKPT_PATH)
            print(f"阶段1训练完成，权重已保存至 {STAGE1_CKPT_PATH}")

# ===================== 阶段2：深度+分割联合训练 =====================
def train_stage2():
    print("\n" + "="*60)
    print("开始阶段2：深度+分割联合训练")
    print("="*60)

    # 1. 初始化模型（继承阶段1权重 + 初始化分割模块）
    ## 1.1 加载阶段1深度模块权重
    netG_A2B = GeneratorA2B().to(DEVICE)
    netD_A = DiscriminatorA().to(DEVICE)
    stage1_ckpt = torch.load(STAGE1_CKPT_PATH, map_location=DEVICE)
    netG_A2B.load_state_dict(stage1_ckpt["netG_A2B"], strict=False)
    netD_A.load_state_dict(stage1_ckpt["netD_A"], strict=False)
    print("成功加载阶段1深度模块权重")

    ## 1.2 初始化双模态骨干
    backbone_rgb = ResNet50(num_classes=None, pretrained=True, in_channels=3).to(DEVICE)
    backbone_depth = ResNet50(num_classes=None, pretrained=False, in_channels=1).to(DEVICE)
    backbone_depth.load_state_dict(backbone_rgb.state_dict(), strict=False)
    # 冻结底层卷积层，仅训练layer4
    for name, param in backbone_rgb.named_parameters():
        if "layer4" not in name:
            param.requires_grad = False
    for name, param in backbone_depth.named_parameters():
        if "layer4" not in name:
            param.requires_grad = False
    print("双模态骨干初始化完成")

    ## 1.3 初始化融合模块和分割头
    fusion_module = CBAM(in_channels=2048).to(DEVICE)
    seg_head = DeepLabV3Plus_Head(in_channels=2048, num_classes=NUM_CLASSES).to(DEVICE)
    init_weights(fusion_module, init_type='normal', init_gain=0.02)
    init_weights(seg_head, init_type='normal', init_gain=0.02)
    print("融合模块和分割头初始化完成")

    ## 1.4 冻结G_B2A（不实例化）
    print("G_B2A已屏蔽，不参与阶段2训练")

    # 2. 定义损失函数和优化器
    criterion_gan = GANLoss(gan_mode='lsgan', device=DEVICE)
    criterion_l1 = DepthL1Loss()
    criterion_seg = SegCrossEntropyLoss(num_classes=NUM_CLASSES)

    # 多模块优化器（分开设置学习率）
    params_G = list(netG_A2B.parameters()) + list(backbone_depth.layer4.parameters())
    params_D = list(netD_A.parameters())
    params_seg = list(fusion_module.parameters()) + list(seg_head.parameters()) + list(backbone_rgb.layer4.parameters())

    optimizer_G = optim.Adam(params_G, lr=LR_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(params_D, lr=LR_D, betas=(0.5, 0.999))
    optimizer_seg = optim.Adam(params_seg, lr=LR_SEG, betas=(0.5, 0.999))

    best_mIoU = 0
    # 3. 开始训练
    for epoch in range(1, STAGE2_EPOCHS+1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_seg_loss = 0.0

        # 批量训练
        pbar = tqdm(range(100), desc=f"Stage2 Epoch {epoch}/{STAGE2_EPOCHS}")
        for _ in pbar:
            # 加载数据
            rgb_img, raw_depth, seg_gt, _ = next(train_dataloader)
            real_depth = raw_depth

            # ----------------------
            # 前向传播
            # ----------------------
            fake_depth = netG_A2B(raw_depth)  # 深度补全
            feat_rgb = backbone_rgb(rgb_img)  # RGB特征提取
            feat_depth = backbone_depth(fake_depth)  # 补全深度特征提取

            # 跨模态融合（拼接+压缩+CBAM）
            fused_input = torch.cat([feat_rgb, feat_depth], dim=1)
            compress_conv = nn.Conv2d(4096, 2048, kernel_size=1, bias=False).to(DEVICE)
            fused_feat = compress_conv(fused_input)
            refined_feat = fusion_module(fused_feat)

            # 语义分割预测
            pred_seg = seg_head(refined_feat)

            # ----------------------
            # 训练生成器 + 分割头
            # ----------------------
            optimizer_G.zero_grad()
            optimizer_seg.zero_grad()

            # 深度对抗损失
            pred_fake = netD_A(fake_depth)
            loss_g_gan = criterion_gan(pred_fake, target_real=True)

            # 深度L1损失
            loss_g_l1 = criterion_l1(fake_depth, real_depth) * LAMBDA_L1

            # 分割损失
            loss_seg = criterion_seg(pred_seg, seg_gt) * LAMBDA_SEG

            # 总损失（生成器+分割头）
            loss_total = loss_g_gan + loss_g_l1 + loss_seg
            loss_total.backward()
            optimizer_G.step()
            optimizer_seg.step()

            # ----------------------
            # 训练判别器
            # ----------------------
            optimizer_D.zero_grad()

            pred_real = netD_A(real_depth)
            loss_d_real = criterion_gan(pred_real, target_real=True)

            fake_depth_detach = fake_depth.detach()
            pred_fake = netD_A(fake_depth_detach)
            loss_d_fake = criterion_gan(pred_fake, target_real=False)

            loss_d_total = (loss_d_real + loss_d_fake) * 0.5
            loss_d_total.backward()
            optimizer_D.step()

            # ----------------------
            # 累计损失
            # ----------------------
            epoch_g_loss += loss_g_gan.item() + loss_g_l1.item()
            epoch_d_loss += loss_d_total.item()
            epoch_seg_loss += loss_seg.item()

            # 更新进度条
            pbar.set_postfix({
                "G Loss": f"{loss_g_gan.item()+loss_g_l1.item():.4f}",
                "D Loss": f"{loss_d_total.item():.4f}",
                "Seg Loss": f"{loss_seg.item():.4f}"
            })

        # ----------------------
        # Epoch结束后处理
        # ----------------------
        avg_g_loss = epoch_g_loss / 100
        avg_d_loss = epoch_d_loss / 100
        avg_seg_loss = epoch_seg_loss / 100
        print(f"Stage2 Epoch {epoch} | Avg G Loss: {avg_g_loss:.4f} | Avg D Loss: {avg_d_loss:.4f} | Avg Seg Loss: {avg_seg_loss:.4f}")

        # 验证
        if epoch % VAL_EVERY_N_EPOCHS == 0:
            avg_mIoU = validate_stage2(netG_A2B, backbone_rgb, backbone_depth, fusion_module, seg_head)
            
            # 6. 新增：保存最优模型
            if avg_mIoU > best_mIoU:
                best_mIoU = avg_mIoU
                torch.save({
                    "epoch": epoch,
                    "netG_A2B": netG_A2B.state_dict(),
                    "netD_A": netD_A.state_dict(),
                    "best_ssim": best_ssim,
                    "optimizer_G": optimizer_G.state_dict()
                }, STAGE1_CKPT_PATH)
                print(f"阶段2最优模型更新！当前最优SSIM: {best_ssim:.4f}，已保存至 {STAGE1_CKPT_PATH}")

        # 保存最终权重
        if epoch == STAGE2_EPOCHS:
            torch.save({
                "epoch": epoch,
                "netG_A2B": netG_A2B.state_dict(),
                "netD_A": netD_A.state_dict(),
                "backbone_rgb": backbone_rgb.state_dict(),
                "backbone_depth": backbone_depth.state_dict(),
                "fusion_module": fusion_module.state_dict(),
                "seg_head": seg_head.state_dict(),
                "avg_seg_loss": avg_seg_loss
            }, STAGE2_CKPT_PATH)
            print(f"阶段2训练完成，权重已保存至 {STAGE2_CKPT_PATH}")

# ===================== 阶段3：全模块联合训练 =====================
def train_stage3():
    print("\n" + "="*60)
    print("开始阶段3：全模块联合训练（深度+分割+图像增强）")
    print("="*60)

    # 1. 初始化模型（继承阶段2权重 + 启用G_B2A/DiscriminatorB）
    ## 1.1 加载阶段2权重
    stage2_ckpt = torch.load(STAGE2_CKPT_PATH, map_location=DEVICE)
    netG_A2B = GeneratorA2B().to(DEVICE)
    netD_A = DiscriminatorA().to(DEVICE)
    backbone_rgb = ResNet50(num_classes=None, pretrained=False, in_channels=3).to(DEVICE)
    backbone_depth = ResNet50(num_classes=None, pretrained=False, in_channels=1).to(DEVICE)
    fusion_module = CBAM(in_channels=2048).to(DEVICE)
    seg_head = DeepLabV3Plus_Head(in_channels=2048, num_classes=NUM_CLASSES).to(DEVICE)

    netG_A2B.load_state_dict(stage2_ckpt["netG_A2B"], strict=False)
    netD_A.load_state_dict(stage2_ckpt["netD_A"], strict=False)
    backbone_rgb.load_state_dict(stage2_ckpt["backbone_rgb"], strict=False)
    backbone_depth.load_state_dict(stage2_ckpt["backbone_depth"], strict=False)
    fusion_module.load_state_dict(stage2_ckpt["fusion_module"], strict=False)
    seg_head.load_state_dict(stage2_ckpt["seg_head"], strict=False)
    print("成功加载阶段2权重")

    ## 1.2 初始化图像增强模块（G_B2A/DiscriminatorB）
    netG_B2A = GeneratorB2A().to(DEVICE)
    netD_B = DiscriminatorB().to(DEVICE)
    init_weights(netG_B2A, init_type='normal', init_gain=0.02)
    init_weights(netD_B, init_type='normal', init_gain=0.02)
    print("图像增强模块初始化完成")

    # 2. 定义损失函数和优化器
    criterion_gan = GANLoss(gan_mode='lsgan', device=DEVICE)
    criterion_l1 = DepthL1Loss()
    criterion_seg = SegCrossEntropyLoss(num_classes=NUM_CLASSES)

    # 全模块参数
    params_all_G = list(netG_A2B.parameters()) + list(netG_B2A.parameters()) + list(backbone_rgb.parameters()) + list(backbone_depth.parameters()) + list(fusion_module.parameters()) + list(seg_head.parameters())
    params_all_D = list(netD_A.parameters()) + list(netD_B.parameters())

    optimizer_G = optim.Adam(params_all_G, lr=LR_G/2, betas=(0.5, 0.999))  # 学习率减半
    optimizer_D = optim.Adam(params_all_D, lr=LR_D/2, betas=(0.5, 0.999))

    # 3. 开始训练
    for epoch in range(1, STAGE3_EPOCHS+1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_cycle_loss = 0.0

        # 批量训练
        pbar = tqdm(range(100), desc=f"Stage3 Epoch {epoch}/{STAGE3_EPOCHS}")
        for _ in pbar:
            # 加载数据
            rgb_img, raw_depth, seg_gt, _ = next(train_dataloader)
            real_depth = raw_depth

            # ----------------------
            # 前向传播（全模块）
            # ----------------------
            fake_depth = netG_A2B(raw_depth)  # 深度补全
            fake_rgb = netG_B2A(rgb_img)      # 图像增强

            # 循环一致性（阶段3核心）
            cycle_depth = netG_A2B(fake_depth)
            cycle_rgb = netG_B2A(fake_rgb)

            # 分割分支前向
            feat_rgb = backbone_rgb(fake_rgb)
            feat_depth = backbone_depth(fake_depth)
            fused_input = torch.cat([feat_rgb, feat_depth], dim=1)
            compress_conv = nn.Conv2d(4096, 2048, kernel_size=1, bias=False).to(DEVICE)
            fused_feat = compress_conv(fused_input)
            refined_feat = fusion_module(fused_feat)
            pred_seg = seg_head(refined_feat)

            # ----------------------
            # 训练生成器（G_A2B+G_B2A）+ 分割头
            # ----------------------
            optimizer_G.zero_grad()

            # 对抗损失（深度+RGB）
            pred_fake_depth = netD_A(fake_depth)
            pred_fake_rgb = netD_B(fake_rgb)
            loss_g_gan_depth = criterion_gan(pred_fake_depth, target_real=True)
            loss_g_gan_rgb = criterion_gan(pred_fake_rgb, target_real=True)

            # 循环一致性损失
            loss_cycle_depth = criterion_l1(cycle_depth, real_depth) * LAMBDA_CYCLE
            loss_cycle_rgb = criterion_l1(cycle_rgb, rgb_img) * LAMBDA_CYCLE

            # 深度L1损失 + 分割损失
            loss_g_l1 = criterion_l1(fake_depth, real_depth) * LAMBDA_L1
            loss_seg = criterion_seg(pred_seg, seg_gt) * LAMBDA_SEG

            # 总生成器损失
            loss_g_total = loss_g_gan_depth + loss_g_gan_rgb + loss_cycle_depth + loss_cycle_rgb + loss_g_l1 + loss_seg
            loss_g_total.backward()
            optimizer_G.step()

            # ----------------------
            # 训练判别器（D_A+D_B）
            # ----------------------
            optimizer_D.zero_grad()

            # 判别器D_A（深度）
            pred_real_depth = netD_A(real_depth)
            loss_d_real_depth = criterion_gan(pred_real_depth, target_real=True)
            pred_fake_depth_detach = netD_A(fake_depth.detach())
            loss_d_fake_depth = criterion_gan(pred_fake_depth_detach, target_real=False)
            loss_d_depth = (loss_d_real_depth + loss_d_fake_depth) * 0.5

            # 判别器D_B（RGB）
            pred_real_rgb = netD_B(rgb_img)
            loss_d_real_rgb = criterion_gan(pred_real_rgb, target_real=True)
            pred_fake_rgb_detach = netD_B(fake_rgb.detach())
            loss_d_fake_rgb = criterion_gan(pred_fake_rgb_detach, target_real=False)
            loss_d_rgb = (loss_d_real_rgb + loss_d_fake_rgb) * 0.5

            # 总判别器损失
            loss_d_total = (loss_d_depth + loss_d_rgb) * 0.5
            loss_d_total.backward()
            optimizer_D.step()

            # ----------------------
            # 累计损失
            # ----------------------
            epoch_g_loss += loss_g_total.item()
            epoch_d_loss += loss_d_total.item()
            epoch_cycle_loss += (loss_cycle_depth.item() + loss_cycle_rgb.item())

            # 更新进度条
            pbar.set_postfix({
                "G Loss": f"{loss_g_total.item():.4f}",
                "D Loss": f"{loss_d_total.item():.4f}",
                "Cycle Loss": f"{loss_cycle_depth.item()+loss_cycle_rgb.item():.4f}"
            })

        # ----------------------
        # Epoch结束后处理
        # ----------------------
        avg_g_loss = epoch_g_loss / 100
        avg_d_loss = epoch_d_loss / 100
        avg_cycle_loss = epoch_cycle_loss / 100
        print(f"Stage3 Epoch {epoch} | Avg G Loss: {avg_g_loss:.4f} | Avg D Loss: {avg_d_loss:.4f} | Avg Cycle Loss: {avg_cycle_loss:.4f}")

        # 保存最终权重
        if epoch == STAGE3_EPOCHS:
            torch.save({
                "epoch": epoch,
                "netG_A2B": netG_A2B.state_dict(),
                "netG_B2A": netG_B2A.state_dict(),
                "netD_A": netD_A.state_dict(),
                "netD_B": netD_B.state_dict(),
                "backbone_rgb": backbone_rgb.state_dict(),
                "backbone_depth": backbone_depth.state_dict(),
                "fusion_module": fusion_module.state_dict(),
                "seg_head": seg_head.state_dict()
            }, STAGE3_CKPT_PATH)
            print(f"阶段3训练完成，权重已保存至 {STAGE3_CKPT_PATH}")

# ===================== 运行三阶段训练 =====================
if __name__ == "__main__":
    # 确保权重保存目录存在
    if not os.path.exists("./"):
        os.makedirs("./")

    # 依次运行三个阶段
    train_stage1()
    train_stage2()
    train_stage3()

    print("\n" + "="*60)
    print("三阶段训练全部完成！")
    print("="*60)