import torch

# 设备配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据配置
IMG_SIZE = 256
NUM_CLASSES = 15  # 语义分割类别数（Cityscapes）
BATCH_SIZE = 4
NUM_WORKERS = 4

# 训练配置
STAGE1_EPOCHS = 50  # 阶段1：深度补全预热
STAGE2_EPOCHS = 100 # 阶段2：深度+分割联合训练
STAGE3_EPOCHS = 150 # 阶段3：全模块联合训练

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