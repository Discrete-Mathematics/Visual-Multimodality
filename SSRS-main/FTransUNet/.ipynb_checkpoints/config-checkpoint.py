import torch

# 设备配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 16

LR_G = 1e-4  # 生成器学习率
LR_D = 1e-4  # 判别器学习率
LR_SEG = 1e-4 # 分割头学习率

# 损失权重配置
LAMBDA_L1 = 100.0  # 深度L1损失权重
LAMBDA_SEG = 50.0  # 分割损失权重
LAMBDA_CYCLE = 10.0 # 循环一致性损失权重（阶段3）

# 权重保存路径
STAGE1_CKPT_PATH = "stage1_depth_pretrain.pth"
STAGE2_CKPT_PATH = "stage2_depth_seg_ckpt.pth"
STAGE3_CKPT_PATH = "stage3_full_model_ckpt.pth"

# 验证集配置
VAL_BATCH_SIZE = 2

# ===================== 配置参数示例 =====================
UNSUPERVISED_CONFIG = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'lambda_cycle': 10.0,    # 循环损失权重（论文伪深度图逻辑延伸）
    'lambda_normal': 5.0,    # 法向量损失权重（几何约束）
    'lambda_crossmodal': 3.0, # 跨模态损失权重（RGB-D融合）
    'lambda_D': 0.5,         # 判别器损失权重（论文WGAN配置）
    'ngf': 64,
    'ndf': 64
}