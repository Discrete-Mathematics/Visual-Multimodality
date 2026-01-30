# ----------------------------------------------------------------------
# 导入所有必要的库
# ----------------------------------------------------------------------
import torch  # 导入PyTorch主库，用于构建和训练神经网络
import torch.nn as nn  # 导入PyTorch的神经网络模块，包含各种层和函数
import torch.optim as optim  # 导入PyTorch的优化器模块，如Adam
from torch.utils.data import Dataset, DataLoader  # 导入数据集和数据加载器相关工具
import torchvision.transforms as transforms  # 导入torchvision的图像变换工具
from PIL import Image  # 导入Pillow库，用于图像的打开、处理和保存
import os  # 导入os库，用于与操作系统交互，如创建文件夹、检查路径
import numpy as np  # 导入NumPy库，主要用于创建伪数据
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1. 定义模型架构 (Model Architecture)
# ----------------------------------------------------------------------
class SSIMLoss(nn.Module):
    """SSIM损失 - 修复版 (增加数值稳定性)"""
    # ... (保持 __init__, gaussian, create_window 不变)
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        window = self.create_window(window_size, self.channel)
        self.register_buffer('window', window)

    # ... (辅助函数省略，保持原样)
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        # 确保输入在合理范围内，防止数值爆炸
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)

        (_, channel, _, _) = img1.size()
        window = self.window.to(img1.device)
        
        if channel != self.channel:
            window = self.create_window(self.window_size, channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2
        
        # 增加 epsilon 防止分母为0
        denominator = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        denominator = torch.clamp(denominator, min=1e-8) 
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / denominator
        ssim_value = torch.clamp(ssim_map.mean(), min=-1.0, max=1.0)
        return 1 - ssim_value
class ColorConstancyLoss(nn.Module):
    """色彩恒常性损失"""
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, enhanced):
        # 计算RGB通道之间的差异
        r = enhanced[:, 0, :, :]
        g = enhanced[:, 1, :, :]
        b = enhanced[:, 2, :, :]
        
        # 色彩恒常性约束：相邻像素的色彩应该平滑变化
        mean_rg = torch.mean(torch.abs(r - g))
        mean_gb = torch.mean(torch.abs(g - b))
        mean_br = torch.mean(torch.abs(b - r))
        
        return (mean_rg + mean_gb + mean_br) / 3.0
class ExposureControlLoss(nn.Module):
    """曝光控制损失"""
    def __init__(self, target_mean=0.5):
        super(ExposureControlLoss, self).__init__()
        self.target_mean = target_mean

    def forward(self, enhanced):
        # 计算图像的平均亮度
        mean_val = torch.mean(enhanced)
        # 目标是将平均亮度控制在合理范围内
        return torch.abs(mean_val - self.target_mean)
class EdgePreservationLoss(nn.Module):
    """边缘保持损失 - 修复版"""
    def __init__(self):
        super(EdgePreservationLoss, self).__init__()
        # ... (保持初始化代码不变)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.epsilon = 1e-6  # 添加极小值防止 sqrt(0)

    def forward(self, enhanced, original):
        sobel_x = self.sobel_x.to(enhanced.device)
        sobel_y = self.sobel_y.to(enhanced.device)
        
        # 原始图像边缘
        original_edges_x = F.conv2d(original, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
        original_edges_y = F.conv2d(original, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
        # 关键修改：在 sqrt 内部加入 epsilon
        original_edges = torch.sqrt(original_edges_x**2 + original_edges_y**2 + self.epsilon)

        # 增强图像边缘
        enhanced_edges_x = F.conv2d(enhanced, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
        enhanced_edges_y = F.conv2d(enhanced, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
        # 关键修改：在 sqrt 内部加入 epsilon
        enhanced_edges = torch.sqrt(enhanced_edges_x**2 + enhanced_edges_y**2 + self.epsilon)

        edge_loss = F.l1_loss(enhanced_edges, original_edges)
        return edge_loss

class FeatureExtractor(nn.Module):
    """
    特征提取器模块。
    与之前版本相同，负责从输入图像中提取高维特征，为门控网络提供决策依据。
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()  # 调用父类nn.Module的构造函数（1）
        # 定义一系列卷积层、激活函数和池化层
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 第一个卷积层：3通道输入，32通道输出（3）
            nn.ReLU(inplace=True),  # ReLU激活函数（4）
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 第二个卷积层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层，尺寸减半（5）
        )#（2）

    def forward(self, x):
        # 定义前向传播逻辑
        return self.conv_block(x)  # 返回提取的特征图（6）

# --- 【核心修改】定义具体的ISP专家 ---
"""
class ExpertGamma(nn.Module):
    def __init__(self):
        super(ExpertGamma, self).__init__() # 调用父类构造函数
        # 定义一个“参数控制器”网络，它接收整张图，输出一个gamma值
        self.controller = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # 卷积层降维(空间降维，通道升维)（7）
            nn.ReLU(), # 激活
            nn.AdaptiveAvgPool2d(1), # 全局平均池化，将特征图变为1x1（8）
            nn.Conv2d(16, 1, kernel_size=1) # 1x1卷积输出一个值
        )

    def forward(self, x):
        # 1. 预测gamma值
        # 控制器网络输出一个值，我们通过Sigmoid将其缩放到(0,1)，然后线性变换到[0.5, 2.5]的合理范围
        gamma = torch.sigmoid(self.controller(x)) * 2.0 + 0.5 # [B, 1, 1, 1]
        
        # 2. 应用伽马校正
        # x.shape = [B, C, H, W], gamma.shape = [B, 1, 1, 1]
        # PyTorch的广播机制会自动将gamma应用到每个像素和每个通道
        return torch.pow(x, gamma) # 返回 图像^gamma（9）
"""
class ExpertGamma(nn.Module):
    def __init__(self, channels=3):
        super(ExpertGamma, self).__init__()
        # 初始化 gamma 为 1.0 (不做改变)
        # 使用 parameter 确保它是可学习的
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        # 1. 钳制 gamma 值
        # gamma 不能小于 0 (否则变成除法)，也不能太大 (导致梯度爆炸)
        # 我们限制在 [0.1, 3.0] 之间，这是合理的图像增强范围
        gamma_safe = torch.clamp(self.gamma, 0.1, 5.0)
        
        # 2. 处理输入 x (解决 NaN 的核心)
        # 因为输入可能是归一化后的负数 (例如 -2.0)，直接求幂会 NaN。
        # 方案 A: 如果原本是 Log 域操作，这里应该 Softplus。
        # 方案 B (推荐): 使用 Softplus 将输入平滑映射到 (0, +inf)
        # 这样既保留了梯度，又保证了 x_pos > 0
        x_pos = F.softplus(x) 
        
        # 3. 安全求幂
        # 加上极小值 epsilon 防止 0 的导数问题
        out = torch.pow(x_pos + 1e-6, gamma_safe)
        
        # 4. (可选) 如果输入原本是负的，通过 Softplus 变正了，
        # 为了保持分布的一致性，我们可以把符号加回去，或者让网络自己适应。
        # 这里既然是增强，我们通常认为 Gamma 是调整亮度的，所以输出正数是合理的。
        
        return out

class ExpertWhiteBalance(nn.Module):
    """
    专家二：自适应白平衡。（11）
    该专家学习为每张图像预测R, G, B三个通道的增益。
    """
    def __init__(self):
        super(ExpertWhiteBalance, self).__init__() # 调用父类构造函数
        # 定义“参数控制器”网络，输出3个通道的增益值
        self.controller = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # 卷积层
            nn.ReLU(), # 激活
            nn.AdaptiveAvgPool2d(1), # 全局平均池化
            nn.Conv2d(16, 3, kernel_size=1) # 1x1卷积输出3个值 (对应R,G,B)
        )

    # 专家可以针对性地利用深度信息
    def forward(self, x):
        # 使用深度信息调整白平衡策略
        depth_aware_gains = torch.sigmoid(self.controller(x)) * 1.5 + 0.5
        
        # 分离RGB和深度通道
        rgb_channels = x[:, :3, :, :]  # [B, 3, H, W]
        depth_channel = x[:, 3:, :, :]  # [B, 1, H, W]
            
        # 只对RGB通道应用增益
        enhanced_rgb = rgb_channels * depth_aware_gains
            
        # 重新组合通道
        return torch.cat([enhanced_rgb, depth_channel], dim=1)

class ExpertContrast(nn.Module):
    """
    专家三：自适应对比度和亮度调整。
    该专家学习为每张图像预测一个对比度因子alpha和亮度偏移beta。
    """
    def __init__(self):
        super(ExpertContrast, self).__init__() # 调用父类构造函数
        # 定义“参数控制器”网络，输出2个值 (alpha和beta)
        self.controller = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # 卷积层
            nn.ReLU(), # 激活
            nn.AdaptiveAvgPool2d(1), # 全局平均池化
            nn.Conv2d(16, 2, kernel_size=1) # 1x1卷积输出2个值
        )

    def forward(self, x):
        # 1. 预测alpha和beta
        params = self.controller(x) # [B, 2, 1, 1]
        # 对比度因子alpha，我们希望它在1附近，范围设为[0.5, 2.0]
        alpha = torch.sigmoid(params[:, 0:1, :, :]) * 1.5 + 0.5 # [B, 1, 1, 1]
        # 亮度偏移beta，我们希望它在0附近，范围设为[-0.2, 0.2]
        beta = (torch.sigmoid(params[:, 1:2, :, :]) - 0.5) * 0.4 # [B, 1, 1, 1]
        
        # 2. 应用对比度和亮度调整
        return x * alpha + beta

class GatingNetwork(nn.Module):
    def __init__(self, input_channels, num_experts):
        """
        修复后的门控网络
        参数:
            input_channels: 输入特征通道数
            num_experts: 专家数量
        """
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts
        # 使用传入的input_channels作为卷积输入通道数
        self.gate_conv = nn.Conv2d(input_channels, num_experts, kernel_size=1)

    def forward(self, features):
        gating_weights = self.gate_conv(features)
        return nn.functional.softmax(gating_weights, dim=1)

class MoE_ISP_Model(nn.Module):
    """
    最终的MoE增强模型，使用具体的ISP专家。
    """

    def __init__(self, num_experts=3, use_depth=True):
        super(MoE_ISP_Model, self).__init__()
        assert num_experts == 3, "此版本的模型固定为3个特定ISP专家"
        self.num_experts = num_experts
        self.use_depth = use_depth  # 控制是否使用深度图
        
        self.feature_extractor = FeatureExtractor()
        
        # 修改门控网络输入通道（32个RGB特征通道 + 1个深度通道）
        gate_input_channels = 33 if use_depth else 32
        self.gating_network = GatingNetwork(gate_input_channels, num_experts)
        
        # 实例化专家（部分专家可能需要调整以接收深度信息）
        self.experts = nn.ModuleList([
            ExpertGamma(),
            ExpertWhiteBalance(), 
            ExpertContrast()
        ])

    def forward(self, x, depth_map=None):
        """
        修改后的前向传播函数，支持深度图作为额外输入
        
        参数:
            x: 输入的低光图像 [B, C, H, W]
            depth_map: 深度图 [B, 1, H, W]，如果为None则使用原始逻辑
        
        返回:
            增强后的图像 [B, C, H, W]
        """
        # 1. 提取RGB特征
        rgb_features = self.feature_extractor(x)  # [B, 32, H/2, W/2]
        
        # 2. 深度图处理与特征融合
        if depth_map is not None and self.use_depth:
            # 确保深度图与RGB图像尺寸匹配
            if depth_map.shape[2:] != x.shape[2:]:
                depth_map = nn.functional.interpolate(depth_map, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            # 提取深度特征（使用简化的卷积处理）
            depth_features = nn.functional.conv2d(
                depth_map, 
                weight=torch.ones(1, 1, 3, 3, device=x.device) / 9.0,  # 平均滤波
                padding=1
            )
            # 下采样深度特征以匹配RGB特征尺寸
            depth_features = nn.functional.interpolate(
                depth_features, 
                size=rgb_features.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            
            # 特征融合：将深度特征与RGB特征拼接
            fused_features = torch.cat([rgb_features, depth_features], dim=1)  # [B, 33, H/2, W/2]
            
            # 调整门控网络输入通道（需要在__init__中预先修改）
            gating_weights = self.gating_network(fused_features)
        else:
            # 回退到原始逻辑（无深度图）
            gating_weights = self.gating_network(rgb_features)
        
        # 3. 权重上采样到原始图像尺寸
        gating_weights = nn.functional.interpolate(
            gating_weights, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # 4. 获取各专家输出（专家也接收深度信息）
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            expert_outputs.append(expert_out)
        
        # 5. 堆叠专家输出并加权融合
        stacked_expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, C, H, W]
        final_output = torch.sum(gating_weights.unsqueeze(2) * stacked_expert_outputs, dim=1)
        
        # 6. 限制输出范围
        return torch.clamp(final_output, 0, 1)

# ----------------------------------------------------------------------
# 2. 数据集处理 (Dataset Handling)
# ----------------------------------------------------------------------
class PairedImageDataset(Dataset):
    def __init__(self, low_light_dir, depth_dir, transform=None):
        self.low_light_dir = low_light_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.low_light_images = sorted(os.listdir(low_light_dir))
        self.depth_images = sorted(os.listdir(depth_dir))
    def __len__(self):
        return len(self.low_light_images)
    def __getitem__(self, idx):
        low_img_name = os.path.join(self.low_light_dir, self.low_light_images[idx])
        depth_img_name = os.path.join(self.depth_dir, self.depth_images[idx])
        low_image = Image.open(low_img_name).convert('RGB')
        depth_image = Image.open(depth_img_name).convert('L')
        if self.transform:
            low_image = self.transform(low_image)
            depth_image = self.transform(depth_image)
        return low_image, depth_image

# ----------------------------------------------------------------------
# 3. 训练过程 (Training Process)
# ----------------------------------------------------------------------
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_dir):
    """
    改进的训练函数：加入验证集验证并保存最佳模型
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 训练设备 (CPU/GPU)
        checkpoint_dir: 模型保存目录
    """
    model.train()
    print("开始训练 (使用具体ISP专家的新模型)...")
    
    # 初始化最佳损失和最佳模型权重
    best_val_loss = float('inf')  # 初始化为正无穷[2,4](@ref)
    best_model_wts = None
    
    # 创建检查点目录
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 记录训练和验证损失
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for i, (low_images, depth_images) in enumerate(train_loader):
            low_images = low_images.to(device)
            depth_images = depth_images.to(device)
            
            optimizer.zero_grad()
            outputs = model(low_images, depth_images)
            loss = criterion(outputs, depth_images)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * low_images.size(0)
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # 计算平均训练损失
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # 验证阶段[4,6](@ref)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for low_images, depth_images in val_loader:
                low_images = low_images.to(device)
                depth_images = depth_images.to(device)
                
                outputs = model(low_images, depth_images)
                loss = criterion(outputs, depth_images)
                val_loss += loss.item() * low_images.size(0)
        
        # 计算平均验证损失
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] 完成, 训练损失: {epoch_train_loss:.4f}, 验证损失: {epoch_val_loss:.4f}")
        
        # 保存最佳模型[1,2](@ref)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = model.state_dict().copy()  # 深拷贝模型参数[10](@ref)
            
            # 保存最佳模型
            best_model_path = os.path.join(checkpoint_dir, 'moe_specific_isp_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': best_val_loss,
            }, best_model_path)
            print(f"✅ 发现新的最佳模型! 验证损失: {best_val_loss:.4f}, 已保存至: {best_model_path}")
        
        # 每5个epoch保存一次检查点[9](@ref)
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"📁 检查点已保存: {checkpoint_path}")
    
    # 训练完成后保存最终模型[9](@ref)
    final_model_path = os.path.join(checkpoint_dir, 'moe_specific_isp_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存至: {final_model_path}")
    
    # 加载最佳模型参数以备后续使用[4](@ref)
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        print(f"已加载最佳模型参数 (验证损失: {best_val_loss:.4f})")
    
    # 返回训练历史记录
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': val_losses.index(min(val_losses)) + 1 if val_losses else 0
    }

def test(model, model_path, test_low_dir, test_depth_dir, output_dir, device):
    """
    改进的测试函数，支持不同格式的图像文件并修复RGBA保存问题
    
    参数:
        model: 模型
        model_path: 模型权重路径
        test_low_dir: 低光图像目录
        test_depth_dir: 深度图目录
        output_dir: 输出目录
        device: 计算设备
    """
    # 加载模型
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("\n开始测试 (使用具体ISP专家的新模型)...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取测试图像列表（仅文件名，不带路径）
    test_images = sorted(os.listdir(test_low_dir))
    
    with torch.no_grad():
        for img_name in test_images:
            # 1. 处理低光图像
            img_path = os.path.join(test_low_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            
            # 2. 处理深度图（根据低光图像文件名生成深度图文件名）
            # 提取文件名（不含扩展名）
            base_name = os.path.splitext(img_name)[0]
            # 构建深度图路径（假设深度图使用.png格式）
            depth_map_path = os.path.join(test_depth_dir, f"{base_name}.png")
            
            # 检查深度图是否存在
            if not os.path.exists(depth_map_path):
                print(f"警告: 深度图不存在 {depth_map_path}，跳过此图像")
                continue
                
            depth_map = Image.open(depth_map_path).convert('L')
            
            # 3. 转换为张量并预处理
            input_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
            depth_tensor = transforms.ToTensor()(depth_map).unsqueeze(0).to(device)
            
            # 4. 模型推理
            enhanced_tensor = model(input_tensor, depth_tensor)
            
            # 5. 保存增强结果
            # 使用原始文件名（保留扩展名）
            save_name = f"enhanced_{img_name}"
            save_path = os.path.join(output_dir, save_name)
            
            # 转换为PIL图像
            enhanced_image_pil = transforms.ToPILImage()(enhanced_tensor.squeeze(0).cpu())
            
            # 关键修复：确保图像是RGB模式（移除Alpha通道）
            if enhanced_image_pil.mode == 'RGBA':
                enhanced_image_pil = enhanced_image_pil.convert('RGB')
            
            # 根据文件扩展名选择合适的保存格式
            file_ext = os.path.splitext(img_name)[1].lower()
            if file_ext in ['.jpg', '.jpeg']:
                enhanced_image_pil.save(save_path, format='JPEG')
            elif file_ext == '.png':
                enhanced_image_pil.save(save_path, format='PNG')
            else:
                # 默认保存为JPEG
                enhanced_image_pil.save(save_path, format='JPEG')
            
            print(f"已增强并保存: {save_path}")

class ImageQualityRegularizer(nn.Module):
    """图像质量正则化器"""
    def __init__(self, weights=None):
        super(ImageQualityRegularizer, self).__init__()
        if weights is None:
            weights = {'ssim': 0.3, 'edge': 0.3, 'color': 0.2, 'exposure': 0.2}
        
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgePreservationLoss()
        self.color_loss = ColorConstancyLoss()
        self.exposure_loss = ExposureControlLoss()
        self.weights = weights
        
    def forward(self, enhanced, original):
        ssim_loss = self.ssim_loss(enhanced, original)
        edge_loss = self.edge_loss(enhanced, original)
        color_loss = self.color_loss(enhanced)
        exposure_loss = self.exposure_loss(enhanced)
        
        total_loss = (self.weights['ssim'] * ssim_loss +
                     self.weights['edge'] * edge_loss +
                     self.weights['color'] * color_loss +
                     self.weights['exposure'] * exposure_loss)
        
        return total_loss, {'ssim': ssim_loss, 'edge': edge_loss, 'color': color_loss, 'exposure': exposure_loss}


# ----------------------------------------------------------------------
# 5. 主执行函数 (Main Execution)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # --- 准备工作：创建虚拟数据和文件夹 ---
    print("正在设置实验环境和创建虚拟数据...") 
    DATA_DIR = "/root/autodl-tmp/new_dataset"#项目目录
    LOW_LIGHT_DIR = os.path.join(DATA_DIR, "train/Image")#训练集低光图像目录
    DEPTH_DIR = os.path.join(DATA_DIR, "train/Depth")#训练集深度图像目录
    VAL_LOW_DIR = os.path.join(DATA_DIR, "val/Image")#验证集低光图像目录
    VAL_DEPTH_DIR = os.path.join(DATA_DIR, "val/Depth")#验证集深度图像目录
    TEST_LOW_DIR = os.path.join(DATA_DIR, "test/Image")#测试集低光图像目录
    TEST_DEPTH_DIR = os.path.join(DATA_DIR, "test/Depth")#测试集深度图像目录
    OUTPUT_DIR = "./enhanced_train" # 使用新的输出文件夹
    CHECKPOINT_DIR = "./checkpoints"#检查点目录
    for path in [LOW_LIGHT_DIR, DEPTH_DIR, VAL_LOW_DIR, VAL_DEPTH_DIR, TEST_LOW_DIR, TEST_DEPTH_DIR, OUTPUT_DIR, CHECKPOINT_DIR]:#创建目录
        os.makedirs(path, exist_ok=True)
    def create_dummy_image(path, size=(128, 128)):
        if not os.listdir(os.path.dirname(path)):
            dummy_array = np.random.randint(0, 256, size + (3,), dtype=np.uint8)
            Image.fromarray(dummy_array, 'RGB').save(path)
    for i in range(5):
        create_dummy_image(os.path.join(LOW_LIGHT_DIR, f"dummy_{i}.png"))
        create_dummy_image(os.path.join(DEPTH_DIR, f"dummy_{i}.png"))
        create_dummy_image(os.path.join(VAL_LOW_DIR, f"dummy_{i}.png"))
        create_dummy_image(os.path.join(VAL_DEPTH_DIR, f"dummy_{i}.png"))
    create_dummy_image(os.path.join(TEST_LOW_DIR, "test_1.png"))
    create_dummy_image(os.path.join(TEST_DEPTH_DIR, "test_1.png"))
    create_dummy_image(os.path.join(TEST_LOW_DIR, "test_2.png"))
    create_dummy_image(os.path.join(TEST_DEPTH_DIR, "test_2.png"))
    print("环境设置完毕。")
    
    # --- 超参数和设置 ---
    NUM_EPOCHS = 5
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"将使用设备: {DEVICE}")

    # --- 训练阶段 ---
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    train_dataset = PairedImageDataset(LOW_LIGHT_DIR, DEPTH_DIR, train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = PairedImageDataset(VAL_LOW_DIR, VAL_DEPTH_DIR, train_transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 实例化新模型 (专家数量固定为3)
    model = MoE_ISP_Model(num_experts=3).to(DEVICE)
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    #train(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE, CHECKPOINT_DIR)

    # --- 测试阶段 ---
    final_model_path = os.path.join("/root/autodl-tmp/checkpoints", 'moe_specific_isp_final.pth') # 加载新模型
    test(model, final_model_path, LOW_LIGHT_DIR, DEPTH_DIR, OUTPUT_DIR, DEVICE)

    print("\n 所有任务完成！")