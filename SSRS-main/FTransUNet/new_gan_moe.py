import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from PIL import Image, ImageFilter

# ---------------------- 补充/修正 导入与全局配置 ----------------------
from config import NUM_CLASSES, LR_G, LR_D, LR_SEG, LAMBDA_L1, LAMBDA_SEG, LAMBDA_CYCLE, UNSUPERVISED_CONFIG
from models.backbone import ResNet50
from models.fusion import CBAM
from models.segmentation import DeepLabV3Plus_Head
from utils.init import init_weights
from utils.losses import DepthL1Loss, SegCrossEntropyLoss, GANLoss
from utils.metrics import (
    calculate_ssim, calculate_gradient_similarity, calculate_phash_similarity,
    calculate_pixel_accuracy, calculate_miou
)
from models.rdfc_gan import UnsupervisedRDFCGAN

# ---------------------- 全局配置（解决未定义变量问题） ----------------------
# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

#训练参数
STAGE1_EPOCHS = 75
STAGE2_EPOCHS = 120
STAGE3_EPOCHS = 105
VAL_EVERY_N_EPOCHS = 5  # 因epochs=1，设为1方便验证
#STAGE1_EPOCHS = STAGE2_EPOCHS = STAGE3_EPOCHS = VAL_EVERY_N_EPOCHS = 1  # 因epochs=1，设为1方便验证
BATCH_SIZE = 2
NUM_WORKERS = 4
MAX_DEPTH_VAL = 65535  # 16-bit深度图归一化值
DISC_TRAIN_STEPS = 10
# 权重保存路径
CKPT_DIR = "./new_new_ckpts"
os.makedirs(CKPT_DIR, exist_ok=True)
STAGE1_CKPT_PATH = os.path.join(CKPT_DIR, "stage1_depth_pretrain.pth")
STAGE2_CKPT_PATH = os.path.join(CKPT_DIR, "stage2_depth_seg_joint.pth")
STAGE3_CKPT_PATH = os.path.join(CKPT_DIR, "stage3_full_module_joint.pth")

UNSUPERVISED_CONFIG = {
    'device': DEVICE,
    'lambda_g': 0.5,        # 论文指定
    'lambda_cycle': 15.0,
    'lambda_normal': 5.0,
    'lambda_crossmodal': 3.0,
    'lambda_D': 0.05,         # 新添加，平衡判别器损失
    'ngf': 64,
    'ndf': 64,
    'lambda_l': 1.0,        # 论文固定
    'lambda_pred': 10.0
}

# ---------------------- 辅助函数：读取txt数据ID ----------------------
def read_txt_to_list(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f'读取文件时出错: {e}')
        return []

# ---------------------- 自定义数据集（修正val/test集缺少max_depth_val问题） ----------------------
class EndToEndDataset(Dataset):
    def __init__(self, ids, rgb_folder, depth_folder, label_folder,
                 target_size=(256, 256), max_depth_val=65535):
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

# ---------------------- 数据加载（修正iter型dataloader耗尽问题） ----------------------
# 数据路径
DATA_ROOT = '/root/autodl-tmp/dataset'
rgb_fmt = os.path.join(DATA_ROOT, 'Image', '{}.jpg')
depth_fmt = os.path.join(DATA_ROOT, 'Depth', '{}.png')
label_fmt = os.path.join(DATA_ROOT, 'Label_Label', '{}_label.png')

# 读取数据ID
train_ids = read_txt_to_list('train.txt')
val_ids = read_txt_to_list('val.txt')
test_ids = read_txt_to_list('test.txt')

# 构建数据集
train_set = EndToEndDataset(
    train_ids, rgb_fmt, depth_fmt, label_fmt,
    target_size=(256, 256),
    max_depth_val=MAX_DEPTH_VAL
)
val_set = EndToEndDataset(
    val_ids, rgb_fmt, depth_fmt, label_fmt,
    target_size=(256, 256),
    max_depth_val=MAX_DEPTH_VAL  # 补充缺失参数
)
test_set = EndToEndDataset(
    test_ids, rgb_fmt, depth_fmt, label_fmt,
    target_size=(256, 256),
    max_depth_val=MAX_DEPTH_VAL  # 补充缺失参数
)

# 构建DataLoader（验证/测试集不使用iter，避免耗尽）
train_dataloader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
)
val_dataloader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
)
test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
)

# ---------------------- 修正验证函数（解决dataloader耗尽+变量对齐） ----------------------
def validate_stage1(depth_model):
    """
    阶段1验证：仅无监督深度补全指标（适配UnsupervisedRDFCGAN）
    """
    print("----- 开始阶段1验证（无监督深度补全）-----")
    depth_model.eval()
    ssim_scores = []
    grad_scores = []
    phash_scores = []
    
    with torch.no_grad():
        for batch_idx, (rgb_img, raw_depth, _, _) in enumerate(val_dataloader):
            if batch_idx >= 50:
                break
            rgb_img, raw_depth = rgb_img.to(DEVICE), raw_depth.to(DEVICE)
            
            # 【关键修复】直接调用 forward，不调用 calculate_loss
            outputs = depth_model(raw_depth, rgb_img)
            fake_depth = outputs['d_pred']
            
            # 计算无监督指标
            ssim_score = calculate_ssim(raw_depth, fake_depth)
            grad_score = calculate_gradient_similarity(raw_depth, fake_depth)
            phash_score = calculate_phash_similarity(raw_depth, fake_depth)
            
            ssim_scores.append(ssim_score)
            grad_scores.append(grad_score)
            phash_scores.append(phash_score)
    
    # 计算平均指标
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    avg_grad = np.mean(grad_scores) if grad_scores else 0.0
    avg_phash = np.mean(phash_scores) if phash_scores else 0.0
    
    print(f"阶段1验证结果 | SSIM: {avg_ssim:.4f} | Grad SIM: {avg_grad:.4f} | pHash SIM: {avg_phash:.4f}")
    depth_model.train()
    return avg_ssim

def validate_stage2(depth_model, backbone_rgb, backbone_depth, fusion_module, seg_head, compress_conv):
    """
    阶段2验证：无监督深度补全 + 有监督语义分割
    """
    print("----- 开始阶段2验证（深度+分割）-----")
    # 切换eval模式
    depth_model.eval()
    backbone_rgb.eval()
    backbone_depth.eval()
    fusion_module.eval()
    seg_head.eval()
    
    # 初始化指标列表
    ssim_scores = []
    grad_scores = []
    pixel_acc_scores = []
    miou_scores = []
    # 压缩卷积移到循环外（避免重复创建）
    #compress_conv = nn.Conv2d(4096, 2048, kernel_size=1, bias=False).to(DEVICE)
    
    with torch.no_grad():
        for batch_idx, (rgb_img, raw_depth, seg_gt, _) in enumerate(val_dataloader):
            if batch_idx >= 50:
                break
            rgb_img, raw_depth, seg_gt = rgb_img.to(DEVICE), raw_depth.to(DEVICE), seg_gt.to(DEVICE)
            # 深度补全前向
            loss_dict = depth_model(raw_depth, rgb_img)
            fake_depth = loss_dict['d_pred']
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
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    avg_grad = np.mean(grad_scores) if grad_scores else 0.0
    avg_pixel_acc = np.mean(pixel_acc_scores) if pixel_acc_scores else 0.0
    avg_miou = np.mean(miou_scores) if miou_scores else 0.0
    
    print(f"阶段2验证结果 | SSIM: {avg_ssim:.4f} | Grad SIM: {avg_grad:.4f} | Pixel Acc: {avg_pixel_acc:.4f} | mIoU: {avg_miou:.4f}")
    # 切换回训练模式
    depth_model.train()
    backbone_rgb.train()
    backbone_depth.train()
    fusion_module.train()
    seg_head.train()
    
    return avg_miou

def validate_stage3(depth_model, img_enhance_model, backbone_rgb, backbone_depth, fusion_module, seg_head, compress_conv):
    """
    阶段3验证：全模块（深度+图像增强+分割）
    """
    print("----- 开始阶段3验证（全模块）-----")
    # 切换eval模式
    depth_model.eval()
    img_enhance_model.eval()
    backbone_rgb.eval()
    backbone_depth.eval()
    fusion_module.eval()
    seg_head.eval()
    
    # 初始化指标列表
    depth_ssim_scores = []
    rgb_ssim_scores = []
    miou_scores = []
    #compress_conv = nn.Conv2d(4096, 2048, kernel_size=1, bias=False).to(DEVICE)
    
    with torch.no_grad():
        for batch_idx, (rgb_img, raw_depth, seg_gt, _) in enumerate(val_dataloader):
            if batch_idx >= 50:
                break
            rgb_img, raw_depth, seg_gt = rgb_img.to(DEVICE), raw_depth.to(DEVICE), seg_gt.to(DEVICE)
            # 深度补全 + 图像增强前向
            loss_dict = depth_model(raw_depth, rgb_img)
            fake_depth = loss_dict['d_pred']
            fake_rgb = img_enhance_model(rgb_img)
            
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
    avg_depth_ssim = np.mean(depth_ssim_scores) if depth_ssim_scores else 0.0
    avg_rgb_ssim = np.mean(rgb_ssim_scores) if rgb_ssim_scores else 0.0
    avg_miou = np.mean(miou_scores) if miou_scores else 0.0
    
    print(f"阶段3验证结果 | Depth SSIM: {avg_depth_ssim:.4f} | RGB SSIM: {avg_rgb_ssim:.4f} | mIoU: {avg_miou:.4f}")
    # 切换回训练模式
    depth_model.train()
    img_enhance_model.train()
    backbone_rgb.train()
    backbone_depth.train()
    fusion_module.train()
    seg_head.train()
    
    return avg_miou

# ---------------------- 阶段1：深度补全预热训练（适配UnsupervisedRDFCGAN） ----------------------
def train_stage1():
    no_improve_epochs = 0
    print("="*60)
    print("开始阶段1：深度补全预热训练")
    print("="*60)

    # 1. 正确初始化无监督深度补全模型（修正原有语法错误）
    depth_model = UnsupervisedRDFCGAN(UNSUPERVISED_CONFIG).to(DEVICE)
    init_weights(depth_model, init_type='normal', init_gain=0.02)

    # 2. 定义优化器（适配UnsupervisedRDFCGAN的生成器/判别器）
    optimizer_gen = optim.Adam(
        list(depth_model.constraint_net.parameters()) + list(depth_model.rdfgan_gen.parameters()),
        lr=LR_G, betas=(0.5, 0.999)
    )
    optimizer_disc = optim.Adam(
        depth_model.rdfgan_disc.parameters(),
        lr=LR_D, betas=(0.5, 0.999)
    )

    # 3. 开始训练
    best_ssim = -1
    
    for epoch in range(1, STAGE1_EPOCHS+1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_cycle_loss = 0.0

        # 批量训练（tqdm进度条，遍历真实dataloader）
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Stage1 Epoch {epoch}/{STAGE1_EPOCHS}")
        for batch_idx, (rgb_img, raw_depth, _, _) in pbar:
            rgb_img, raw_depth = rgb_img.to(DEVICE), raw_depth.to(DEVICE)
            # ----------------------
            # 训练生成器（约束网络+RDF-GAN生成器）
            # ----------------------
            optimizer_gen.zero_grad()
            #print([raw_depth.shape, rgb_img.shape])
            loss_dict = depth_model.calculate_loss(raw_depth, rgb_img) # raw_depth:[8, 1, 256, 256] rgb_img:[8, 3, 256, 256]
            loss_g_total = loss_dict['L_overall']
            fake_depth = loss_dict['d_pred']
            loss_cycle = loss_dict.get('L_cycle', torch.tensor(0.0).to(DEVICE))  # 兼容无监督循环损失
            loss_g_total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(depth_model.constraint_net.parameters()) + list(depth_model.rdfgan_gen.parameters()),
                max_norm=1.0
            )
            optimizer_gen.step()

            # ----------------------
            # 训练判别器（RDF-GAN判别器）
            # ----------------------
            batch_d_loss = 0.0
            for _ in range(DISC_TRAIN_STEPS):
                optimizer_disc.zero_grad()  # 每次训练前清零梯度
                loss_dict = depth_model.calculate_loss(raw_depth, rgb_img)
                loss_d_total = loss_dict['L_D']
                loss_d_total.backward()  # 反向传播
                torch.nn.utils.clip_grad_norm_(depth_model.rdfgan_disc.parameters(), max_norm=1)
                optimizer_disc.step()  # 更新判别器权重
                
                batch_d_loss += loss_d_total.item()  # 累加当前批次内判别器的多次训练损失

            # ----------------------
            # 累计损失（修正：判别器损失取批次内多次训练的平均值）
            # ----------------------
            epoch_g_loss += loss_g_total.item()
            epoch_d_loss += (batch_d_loss / DISC_TRAIN_STEPS)  # 取平均，保证epoch平均损失准确()
            epoch_cycle_loss += loss_cycle.item() if loss_cycle is not None else 0.0

            # 更新进度条
            pbar.set_postfix({
                "G Loss": f"{loss_g_total.item():.4f}",
                "D Loss": f"{loss_d_total.item():.4f}",
                "Cycle Loss": f"{loss_cycle.item():.4f}" if loss_cycle is not None else "0.0000"
            })

        # ----------------------
        # 每个Epoch结束后：打印平均损失、保存权重
        # ----------------------
        avg_g_loss = epoch_g_loss / len(train_dataloader)
        avg_d_loss = epoch_d_loss / len(train_dataloader)
        avg_cycle_loss = epoch_cycle_loss / len(train_dataloader)
        print(f"Stage1 Epoch {epoch} | Avg G Loss: {avg_g_loss:.4f} | Avg D Loss: {avg_d_loss:.4f} | Avg Cycle Loss: {avg_cycle_loss:.4f}")

        # 验证

        if epoch % VAL_EVERY_N_EPOCHS == 0:
            current_ssim = validate_stage1(depth_model)
            
            # 保存最优模型
            if current_ssim > best_ssim:
                best_ssim = current_ssim
                no_improve_epochs = 0  # 重置计数
                torch.save({
                    "epoch": epoch,
                    "depth_model": depth_model.state_dict(),
                    "best_ssim": best_ssim,
                    "optimizer_gen": optimizer_gen.state_dict(),
                    "optimizer_disc": optimizer_disc.state_dict()
                }, STAGE1_CKPT_PATH)
                print(f"阶段1最优模型更新！当前最优SSIM: {best_ssim:.4f}，已保存至 {STAGE1_CKPT_PATH}")
            else:
                no_improve_epochs += 1
                print(f"SSIM连续{no_improve_epochs}个验证周期未提升，当前最优：{best_ssim:.4f}")
                # 连续3个周期不提升，提前终止训练（避免恶化）
                if no_improve_epochs >= 3:
                    print("SSIM连续3个验证周期未提升，提前终止Stage1训练")
                    return  # 退出train_stage1函数
                
        # 保存最终权重
        if epoch == STAGE1_EPOCHS:
            torch.save({
                "epoch": epoch,
                "depth_model": depth_model.state_dict(),
                "optimizer_gen": optimizer_gen.state_dict(),
                "optimizer_disc": optimizer_disc.state_dict(),
                "avg_cycle_loss": avg_cycle_loss
            }, STAGE1_CKPT_PATH)
            print(f"阶段1训练完成，权重已保存至 {STAGE1_CKPT_PATH}")

# ---------------------- 阶段2：深度+分割联合训练 ----------------------
def train_stage2():
    no_improve_epochs = 0
    print("\n" + "="*60)
    print("开始阶段2：深度+分割联合训练")
    print("="*60)

    # 1. 初始化模型（继承阶段1权重 + 初始化分割模块）
    ## 1.1 加载阶段1深度模型权重
    depth_model = UnsupervisedRDFCGAN(UNSUPERVISED_CONFIG).to(DEVICE)
    stage1_ckpt = torch.load(STAGE1_CKPT_PATH, map_location=DEVICE, weights_only=False)
    depth_model.load_state_dict(stage1_ckpt["depth_model"], strict=False)
    print("成功加载阶段1深度模块权重")
    ## 1.2 初始化双模态骨干
    backbone_rgb = ResNet50(num_classes=None, pretrained=True, in_channels=3).to(DEVICE)
    backbone_depth = ResNet50(num_classes=None, pretrained=False, in_channels=1).to(DEVICE)
    
    # 【关键修复】加载RGB预训练权重到Depth骨干，但跳过第一层卷积（输入通道3 vs 1不匹配）
    rgb_state_dict = backbone_rgb.state_dict()
    # 过滤掉conv1相关的所有权重（conv1.weight, conv1.bias等）
    filtered_state_dict = {k: v for k, v in rgb_state_dict.items() if not k.startswith('conv1')}
    backbone_depth.load_state_dict(filtered_state_dict, strict=False)
    print("双模态骨干初始化完成（Depth骨干第一层随机初始化，其余层继承RGB预训练权重）")

    # 可选：将RGB预训练权重的3个通道平均作为单通道的初始权重
    rgb_conv1_weight = rgb_state_dict['conv1.weight']  # [64, 3, 7, 7]
    depth_conv1_weight = rgb_conv1_weight.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
    depth_state_dict = backbone_depth.state_dict()
    depth_state_dict['conv1.weight'] = depth_conv1_weight
    backbone_depth.load_state_dict(depth_state_dict)

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

    ## 1.4 压缩卷积（移到循环外，避免重复创建）
    compress_conv = nn.Conv2d(4096, 2048, kernel_size=1, bias=False).to(DEVICE)

    # 2. 定义损失函数和优化器
    criterion_gan = GANLoss(gan_mode='lsgan', device=DEVICE)
    criterion_l1 = DepthL1Loss()
    criterion_seg = SegCrossEntropyLoss(num_classes=NUM_CLASSES)

    # 多模块优化器（分开设置学习率）
    params_depth = list(depth_model.parameters()) + list(backbone_depth.layer4.parameters())
    params_disc = list(depth_model.rdfgan_disc.parameters())
    params_seg = list(fusion_module.parameters()) + list(seg_head.parameters()) + list(backbone_rgb.layer4.parameters())

    optimizer_depth = optim.Adam(params_depth, lr=LR_G, betas=(0.5, 0.999))
    optimizer_disc = optim.Adam(params_disc, lr=LR_D, betas=(0.5, 0.999))
    optimizer_seg = optim.Adam(params_seg, lr=LR_SEG, betas=(0.5, 0.999))

    # 3. 开始训练
    best_mIoU = 0.0
    
    for epoch in range(1, STAGE2_EPOCHS+1):
        epoch_depth_loss = 0.0
        epoch_disc_loss = 0.0
        epoch_seg_loss = 0.0

        # 批量训练
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Stage2 Epoch {epoch}/{STAGE2_EPOCHS}")
        for batch_idx, (rgb_img, raw_depth, seg_gt, _) in pbar:
            rgb_img, raw_depth, seg_gt = rgb_img.to(DEVICE), raw_depth.to(DEVICE), seg_gt.to(DEVICE)
            # ----------------------
            # 前向传播
            # ----------------------
            loss_dict = depth_model.calculate_loss(raw_depth, rgb_img)  # 深度补全
            fake_depth = loss_dict['d_pred']
            feat_rgb = backbone_rgb(rgb_img)  # RGB特征提取
            feat_depth = backbone_depth(fake_depth)  # 补全深度特征提取

            # 跨模态融合（拼接+压缩+CBAM）
            fused_input = torch.cat([feat_rgb, feat_depth], dim=1)
            fused_feat = compress_conv(fused_input)
            refined_feat = fusion_module(fused_feat)
            pred_seg = seg_head(refined_feat)

            # ----------------------
            # 训练深度模块 + 分割头
            # ----------------------
            optimizer_depth.zero_grad()
            optimizer_seg.zero_grad()

            # 深度损失（无监督+L1）
            loss_depth_l1 = criterion_l1(fake_depth, raw_depth) * LAMBDA_L1
            loss_gan = criterion_gan(depth_model.rdfgan_disc(fake_depth, rgb_img), target_real=True)

            # 分割损失
            loss_seg = criterion_seg(pred_seg, seg_gt) * LAMBDA_SEG

            # 总损失
            loss_total = loss_depth_l1 + loss_gan + loss_seg
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(params_depth+params_seg, max_norm=1.0)
            optimizer_depth.step()
            optimizer_seg.step()

            # ----------------------
            # 训练判别器
            # ----------------------
            batch_d_loss = 0.0  # 记录当前批次判别器多次训练的总损失
            for _ in range(DISC_TRAIN_STEPS):
                optimizer_disc.zero_grad()  # 每次训练前清零梯度

                pred_real = depth_model.rdfgan_disc(raw_depth, rgb_img)
                loss_d_real = criterion_gan(pred_real, target_real=True)

                fake_depth_detach = fake_depth.detach()  # 阻断梯度回传至生成器
                pred_fake = depth_model.rdfgan_disc(fake_depth_detach, rgb_img)
                loss_d_fake = criterion_gan(pred_fake, target_real=False)

                loss_d_total = (loss_d_real + loss_d_fake) * 0.5
                loss_d_total.backward()
                torch.nn.utils.clip_grad_norm_(params_disc, max_norm=1.0)
                optimizer_disc.step()

                batch_d_loss += loss_d_total.item()  # 累加批次内多次训练损失

            # ----------------------
            # 累计损失（修正：判别器损失取批次内平均值）
            # ----------------------
            epoch_depth_loss += (loss_depth_l1.item() + loss_gan.item())
            epoch_disc_loss += (batch_d_loss / DISC_TRAIN_STEPS)  # 取平均保证epoch损失准确
            epoch_seg_loss += loss_seg.item()

            # 更新进度条
            pbar.set_postfix({
                "Depth Loss": f"{loss_depth_l1.item()+loss_gan.item():.4f}",
                "D Loss": f"{loss_d_total.item():.4f}",
                "Seg Loss": f"{loss_seg.item():.4f}"
            })

        # ----------------------
        # Epoch结束后处理
        # ----------------------
        avg_depth_loss = epoch_depth_loss / len(train_dataloader)
        avg_disc_loss = epoch_disc_loss / len(train_dataloader)
        avg_seg_loss = epoch_seg_loss / len(train_dataloader)
        print(f"Stage2 Epoch {epoch} | Avg Depth Loss: {avg_depth_loss:.4f} | Avg D Loss: {avg_disc_loss:.4f} | Avg Seg Loss: {avg_seg_loss:.4f}")

        # 验证
        if epoch % VAL_EVERY_N_EPOCHS == 0:
            avg_mIoU = validate_stage2(depth_model, backbone_rgb, backbone_depth, fusion_module, seg_head, compress_conv)
            
            # 保存最优模型（修正路径+变量错误）
            if avg_mIoU > best_mIoU:
                best_mIoU = avg_mIoU
                no_improve_epochs = 0
                torch.save({
                    "epoch": epoch,
                    "depth_model": depth_model.state_dict(),
                    "backbone_rgb": backbone_rgb.state_dict(),
                    "backbone_depth": backbone_depth.state_dict(),
                    "fusion_module": fusion_module.state_dict(),
                    "seg_head": seg_head.state_dict(),
                    "best_mIoU": best_mIoU
                }, STAGE2_CKPT_PATH)
                print(f"阶段2最优模型更新！当前最优mIoU: {best_mIoU:.4f}，已保存至 {STAGE2_CKPT_PATH}")
            else:
                no_improve_epochs += 1
                print(f"avg mIoU连续{no_improve_epochs}个验证周期未提升，当前最优：{best_mIoU:.4f}")
                # 连续3个周期不提升，提前终止训练（避免恶化）
                if no_improve_epochs >= 3:
                    print("avg mIoU连续3个验证周期未提升，提前终止Stage2训练")
                    return  # 退出train_stage2函数
                    
        # 保存最终权重
        if epoch == STAGE2_EPOCHS:
            torch.save({
                "epoch": epoch,
                "depth_model": depth_model.state_dict(),
                "backbone_rgb": backbone_rgb.state_dict(),
                "backbone_depth": backbone_depth.state_dict(),
                "fusion_module": fusion_module.state_dict(),
                "seg_head": seg_head.state_dict(),
                "avg_seg_loss": avg_seg_loss
            }, STAGE2_CKPT_PATH)
            print(f"阶段2训练完成，权重已保存至 {STAGE2_CKPT_PATH}")

# ---------------------- 阶段3：全模块联合训练 ----------------------
def train_stage3():
    no_improve_epochs = 0
    print("\n" + "="*60)
    print("开始阶段3：全模块联合训练（深度+分割+图像增强）")
    print("="*60)

    # 1. 初始化模型（继承阶段2权重 + 启用图像增强模块）
    ## 1.1 加载阶段2权重
    stage2_ckpt = torch.load(STAGE2_CKPT_PATH, map_location=DEVICE, weights_only=False)
    depth_model = UnsupervisedRDFCGAN(UNSUPERVISED_CONFIG).to(DEVICE)
    backbone_rgb = ResNet50(num_classes=None, pretrained=False, in_channels=3).to(DEVICE)
    backbone_depth = ResNet50(num_classes=None, pretrained=False, in_channels=1).to(DEVICE)
    fusion_module = CBAM(in_channels=2048).to(DEVICE)
    seg_head = DeepLabV3Plus_Head(in_channels=2048, num_classes=NUM_CLASSES).to(DEVICE)

    depth_model.load_state_dict(stage2_ckpt["depth_model"], strict=False)
    backbone_rgb.load_state_dict(stage2_ckpt["backbone_rgb"], strict=False)
    backbone_depth.load_state_dict(stage2_ckpt["backbone_depth"], strict=False)
    fusion_module.load_state_dict(stage2_ckpt["fusion_module"], strict=False)
    seg_head.load_state_dict(stage2_ckpt["seg_head"], strict=False)
    print("成功加载阶段2权重")

    ## 1.2 初始化图像增强模块（G_B2A，对齐UnsupervisedRDFCGAN架构）
    from models.rdfc_gan import GeneratorB2A  # 补充图像增强生成器导入
    img_enhance_model = GeneratorB2A().to(DEVICE)
    init_weights(img_enhance_model, init_type='normal', init_gain=0.02)
    print("图像增强模块初始化完成")

    ## 1.3 压缩卷积
    compress_conv = nn.Conv2d(4096, 2048, kernel_size=1, bias=False).to(DEVICE)

    # 2. 定义损失函数和优化器（学习率减半）
    criterion_gan = GANLoss(gan_mode='lsgan', device=DEVICE)
    criterion_l1 = DepthL1Loss()
    criterion_seg = SegCrossEntropyLoss(num_classes=NUM_CLASSES)

    # 全模块参数
    params_all_G = list(depth_model.parameters()) + list(img_enhance_model.parameters()) + \
                   list(backbone_rgb.parameters()) + list(backbone_depth.parameters()) + \
                   list(fusion_module.parameters()) + list(seg_head.parameters())
    params_all_D = list(depth_model.rdfgan_disc.parameters())

    optimizer_G = optim.Adam(params_all_G, lr=LR_G/2, betas=(0.5, 0.999))  # 学习率减半
    optimizer_D = optim.Adam(params_all_D, lr=LR_D/2, betas=(0.5, 0.999))

    # 3. 开始训练
    for epoch in range(1, STAGE3_EPOCHS+1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_cycle_loss = 0.0

        # 批量训练
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Stage3 Epoch {epoch}/{STAGE3_EPOCHS}")
        for batch_idx, (rgb_img, raw_depth, seg_gt, _) in pbar:
            rgb_img, raw_depth, seg_gt = rgb_img.to(DEVICE), raw_depth.to(DEVICE), seg_gt.to(DEVICE)
            # ----------------------
            # 前向传播（全模块）
            # ----------------------
            loss_dict = depth_model.calculate_loss(raw_depth, rgb_img)  # 深度补全
            fake_depth = loss_dict['d_pred']
            fake_rgb = img_enhance_model(rgb_img)      # 图像增强

            # 循环一致性（阶段3核心）
            loss_dict = depth_model.calculate_loss(fake_depth, fake_rgb)
            cycle_depth = loss_dict['d_pred']
            cycle_rgb = img_enhance_model(fake_rgb)

            # 分割分支前向
            feat_rgb = backbone_rgb(fake_rgb)
            feat_depth = backbone_depth(fake_depth)
            fused_input = torch.cat([feat_rgb, feat_depth], dim=1)
            fused_feat = compress_conv(fused_input)
            refined_feat = fusion_module(fused_feat)
            pred_seg = seg_head(refined_feat)

            # ----------------------
            # 训练生成器（深度+图像增强）+ 分割头
            # ----------------------
            optimizer_G.zero_grad()

            # 对抗损失（深度）
            pred_fake_depth = depth_model.rdfgan_disc(fake_depth, rgb_img)
            loss_g_gan_depth = criterion_gan(pred_fake_depth, target_real=True)

            # 循环一致性损失
            loss_cycle_depth = criterion_l1(cycle_depth, raw_depth) * LAMBDA_CYCLE
            loss_cycle_rgb = criterion_l1(cycle_rgb, rgb_img) * LAMBDA_CYCLE

            # 深度L1损失 + 分割损失
            loss_g_l1 = criterion_l1(fake_depth, raw_depth) * LAMBDA_L1
            loss_seg = criterion_seg(pred_seg, seg_gt) * LAMBDA_SEG

            # 总生成器损失
            loss_g_total = loss_g_gan_depth + loss_cycle_depth + loss_cycle_rgb + loss_g_l1 + loss_seg
            loss_g_total.backward()
            torch.nn.utils.clip_grad_norm_(params_all_G, max_norm=1.0)
            optimizer_G.step()

            # ----------------------
            # 训练判别器（深度）
            # ----------------------
            batch_d_loss = 0.0  # 记录当前批次判别器多次训练的总损失
            for _ in range(DISC_TRAIN_STEPS):
                optimizer_D.zero_grad()  # 每次训练前清零梯度

                # 判别器D_A（深度）
                pred_real_depth = depth_model.rdfgan_disc(raw_depth, rgb_img)
                loss_d_real_depth = criterion_gan(pred_real_depth, target_real=True)
                pred_fake_depth_detach = depth_model.rdfgan_disc(fake_depth.detach(), rgb_img)
                loss_d_fake_depth = criterion_gan(pred_fake_depth_detach, target_real=False)
                loss_d_total = (loss_d_real_depth + loss_d_fake_depth) * 0.5

                loss_d_total.backward()
                torch.nn.utils.clip_grad_norm_(params_all_D, max_norm=1.0)
                optimizer_D.step()

                batch_d_loss += loss_d_total.item()  # 累加批次内多次训练损失

            # ----------------------
            # 累计损失（修正：判别器损失取批次内平均值）
            # ----------------------
            epoch_g_loss += loss_g_total.item()
            epoch_d_loss += (batch_d_loss / DISC_TRAIN_STEPS)  # 取平均保证epoch损失准确
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
        avg_g_loss = epoch_g_loss / len(train_dataloader)
        avg_d_loss = epoch_d_loss / len(train_dataloader)
        avg_cycle_loss = epoch_cycle_loss / len(train_dataloader)
        print(f"Stage3 Epoch {epoch} | Avg G Loss: {avg_g_loss:.4f} | Avg D Loss: {avg_d_loss:.4f} | Avg Cycle Loss: {avg_cycle_loss:.4f}")

        if epoch % VAL_EVERY_N_EPOCHS == 0:
            avg_mIoU = validate_stage3(depth_model, backbone_rgb, backbone_depth, fusion_module, seg_head, compress_conv)
            
            # 保存最优模型（修正路径+变量错误）
            if avg_mIoU > best_mIoU:
                best_mIoU = avg_mIoU
                no_improve_epochs = 0
                torch.save({
                    "epoch": epoch,
                    "depth_model": depth_model.state_dict(),
                    "backbone_rgb": backbone_rgb.state_dict(),
                    "backbone_depth": backbone_depth.state_dict(),
                    "fusion_module": fusion_module.state_dict(),
                    "seg_head": seg_head.state_dict(),
                    "best_mIoU": best_mIoU
                }, STAGE3_CKPT_PATH)
                print(f"阶段3最优模型更新！当前最优mIoU: {best_mIoU:.4f}，已保存至 {STAGE3_CKPT_PATH}")
            else:
                no_improve_epochs += 1
                print(f"avg mIoU连续{no_improve_epochs}个验证周期未提升，当前最优：{best_mIoU:.4f}")
                # 连续3个周期不提升，提前终止训练（避免恶化）
                if no_improve_epochs >= 3:
                    print("avg mIoU连续3个验证周期未提升，提前终止Stage3训练")
                    return  # 退出train_stage3函数
                    
        # 保存最终权重
        if epoch == STAGE3_EPOCHS:
            torch.save({
                "epoch": epoch,
                "depth_model": depth_model.state_dict(),
                "img_enhance_model": img_enhance_model.state_dict(),
                "backbone_rgb": backbone_rgb.state_dict(),
                "backbone_depth": backbone_depth.state_dict(),
                "fusion_module": fusion_module.state_dict(),
                "seg_head": seg_head.state_dict()
            }, STAGE3_CKPT_PATH)
            print(f"阶段3训练完成，权重已保存至 {STAGE3_CKPT_PATH}")

# ---------------------- 运行三阶段训练 ----------------------
if __name__ == "__main__":
    # 确保权重保存目录存在
    os.makedirs(CKPT_DIR, exist_ok=True)

    # 依次运行三个阶段
    #train_stage1()
    train_stage2()
    train_stage3()

    print("\n" + "="*60)
    print("三阶段训练全部完成！")
    print("="*60)