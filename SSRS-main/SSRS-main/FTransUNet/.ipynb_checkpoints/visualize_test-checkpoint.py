import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image
import json
from tqdm import tqdm
import argparse

# 导入你项目中的必要模块
from enhancement import MoE_ISP_Model
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import torchvision.transforms as transforms


plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# ==================== 配置参数 ====================
class VisualizationConfig:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 4  # 较小的批次大小以便更好地可视化
        self.num_samples_to_visualize = 20  # 要可视化的样本数量
        self.save_individual_images = True  # 是否保存单个图像
        self.create_comparison_grid = True  # 是否创建对比网格
        self.output_dir = "./visualization_results"
        
        # 语义分割类别颜色映射 (根据你的16个类别自定义)
        self.class_colors = [
            [0, 0, 0],           # 0: background - 黑色
            [128, 0, 0],         # 1: person - 红色
            [0, 128, 0],         # 2: cable - 深绿色
            [128, 128, 0],       # 3: tube - 橄榄色
            [0, 0, 128],         # 4: indicator - 深蓝色
            [128, 0, 128],       # 5: electrical equipment - 紫色
            [0, 128, 128],       # 6: electronic equipment - 青色
            [128, 128, 128],     # 7: mining equipment - 灰色
            [64, 0, 0],          # 8: rail area - 棕红色
            [128, 64, 0],        # 9: support equipment - 橙色
            [64, 128, 0],        # 10: door - 黄绿色
            [192, 128, 0],       # 11: tools and materials - 橙色
            [64, 0, 128],        # 12: rescue equipment - 紫红色
            [192, 0, 128],       # 13: container - 粉紫色
            [64, 128, 128],      # 14: metal fixture - 蓝绿色
            [192, 128, 128],     # 15: anchoring equipment - 粉红色
        ]
        
        self.class_names = [
            "背景", "类别1", "类别2", "类别3", "类别4", "类别5", 
            "类别6", "类别7", "类别8", "类别9", "类别10", 
            "类别11", "类别12", "类别13", "类别14", "类别15"
        ]

# ==================== 数据加载器 ====================
class VisualizationDataset(torch.utils.data.Dataset):
    def __init__(self, ids, image_folder, depth_folder, label_folder, transform=None):
        self.ids = ids
        self.image_folder = image_folder
        self.depth_folder = depth_folder
        self.label_folder = label_folder
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        
        # 加载图像
        img = Image.open(self.image_folder.format(id)).convert('RGB')
        original_size = img.size  # 保存原始尺寸
        
        # 加载深度图
        dsm = Image.open(self.depth_folder.format(id)).convert('L')
        
        # 加载标签
        label = Image.open(self.label_folder.format(id)).convert('L')
        
        # 转换为张量
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
            
        dsm_tensor = torch.tensor(np.array(dsm), dtype=torch.float32).unsqueeze(0) / 255.0
        label_tensor = torch.tensor(np.array(label), dtype=torch.long)
        
        return {
            'id': id,
            'image': img_tensor,
            'depth': dsm_tensor,
            'label': label_tensor,
            'original_image': np.array(img),
            'original_size': original_size
        }

# ==================== 模型加载 ====================
def load_best_model(config_path, model_path, device):
    """加载最佳模型"""
    print("🚀 加载最佳模型...")
    
    # 加载配置
    with open(config_path, 'r') as f:
        experiment_config = json.load(f)
    
    # 初始化MoE模型
    moe_model = MoE_ISP_Model(num_experts=3)
    moe_model = moe_model.to(device)
    
    # 初始化分割模型
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 16
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(256 / 16), int(256 / 16))
    
    ftransunet_model = ViT_seg(config_vit, img_size=256, num_classes=16)
    ftransunet_model = ftransunet_model.to(device)
    
    # 组合端到端模型
    class EndToEndModel(nn.Module):
        def __init__(self, moe_model, ftransunet_model):
            super(EndToEndModel, self).__init__()
            self.moe_enhancer = moe_model
            self.ftransunet_segmenter = ftransunet_model
            
        def forward(self, x, depth_map):
            enhanced_img = self.moe_enhancer(x, depth_map)
            enhanced_img = torch.clamp(enhanced_img, 0.0, 1.0)
            segmentation_output = self.ftransunet_segmenter(enhanced_img, depth_map.squeeze(1))
            return enhanced_img, segmentation_output
    
    model = EndToEndModel(moe_model, ftransunet_model)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 模型加载成功 (Epoch: {checkpoint.get('epoch', 'N/A')}, "
          f"mIoU: {checkpoint.get('best_mIoU', 'N/A'):.4f})")
    
    return model, experiment_config

# ==================== 可视化函数 ====================
def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反归一化图像用于显示"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def create_segmentation_color_map(prediction, class_colors):
    """将预测结果转换为彩色图像"""
    h, w = prediction.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in enumerate(class_colors):
        mask = prediction == class_id
        colored[mask] = color
    
    return colored

def visualize_single_sample(sample, prediction, enhanced_img, config, save_path):
    """可视化单个样本的完整结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"prediction - ID: {sample['id']}", fontsize=16, fontweight='bold')
    
    # 原始图像
    original_img = denormalize_image(sample['image']).permute(1, 2, 0).numpy()
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('RGB', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 深度图
    depth_map = sample['depth'].squeeze().numpy()
    axes[0, 1].imshow(depth_map, cmap='viridis')
    axes[0, 1].set_title('depth', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 增强后图像
    enhanced_img_np = denormalize_image(enhanced_img).permute(1, 2, 0).numpy()
    axes[0, 2].imshow(enhanced_img_np)
    axes[0, 2].set_title('enhanced', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 真实标签
    true_label = sample['label'].numpy()
    true_colored = create_segmentation_color_map(true_label, config.class_colors)
    axes[1, 0].imshow(true_colored)
    axes[1, 0].set_title('gt', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 预测结果
    pred_colored = create_segmentation_color_map(prediction, config.class_colors)
    axes[1, 1].imshow(pred_colored)
    axes[1, 1].set_title('pred', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 误差图（红色显示错误分类的区域）
    error_mask = (true_label != prediction) & (true_label != 255)  # 忽略无效标签
    error_visualization = true_colored.copy()
    error_visualization[error_mask] = [255, 0, 0]  # 红色标记错误
    
    axes[1, 2].imshow(error_visualization)
    axes[1, 2].set_title('wrong_region(red)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 保存单个样本可视化: {save_path}")

def create_class_distribution_chart(predictions, true_labels, config, save_path):
    """创建类别分布对比图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 计算预测和真实标签的类别分布
    pred_counts = np.bincount(predictions.ravel(), minlength=len(config.class_names))
    true_counts = np.bincount(true_labels.ravel(), minlength=len(config.class_names))
    
    # 忽略背景类别（索引0）如果它占主导地位
    show_classes = range(1, min(10, len(config.class_names)))  # 只显示前9个非背景类别
    
    # 预测分布
    colors_pred = [np.array(config.class_colors[i])/255.0 for i in show_classes]
    ax1.bar(range(len(show_classes)), [pred_counts[i] for i in show_classes], 
            color=colors_pred, alpha=0.7)
    ax1.set_title('predicted_cls_distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(show_classes)))
    ax1.set_xticklabels([config.class_names[i] for i in show_classes], rotation=45)
    ax1.set_ylabel('pixel_num')
    
    # 真实分布
    colors_true = [np.array(config.class_colors[i])/255.0 for i in show_classes]
    ax2.bar(range(len(show_classes)), [true_counts[i] for i in show_classes], 
            color=colors_true, alpha=0.7)
    ax2.set_title('gt_distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(show_classes)))
    ax2.set_xticklabels([config.class_names[i] for i in show_classes], rotation=45)
    ax2.set_ylabel('pixel_num')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存类别分布图: {save_path}")

def create_comparison_grid(samples, predictions, enhanced_imgs, config, save_path, grid_size=(4, 5)):
    """创建样本对比网格"""
    n_samples = min(len(samples), grid_size[0] * grid_size[1])
    fig, axes = plt.subplots(grid_size[0], grid_size[1] * 3, figsize=(20, 12))
    
    if grid_size[0] == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        row = i // grid_size[1]
        col_base = (i % grid_size[1]) * 3
        
        # 原始图像
        original_img = denormalize_image(samples[i]['image']).permute(1, 2, 0).numpy()
        axes[row, col_base].imshow(original_img)
        if i == 0:
            axes[row, col_base].set_title('RGB', fontsize=10)
        axes[row, col_base].axis('off')
        
        # 真实标签
        true_label = samples[i]['label'].numpy()
        true_colored = create_segmentation_color_map(true_label, config.class_colors)
        axes[row, col_base + 1].imshow(true_colored)
        if i == 0:
            axes[row, col_base + 1].set_title('ground_truth', fontsize=10)
        axes[row, col_base + 1].axis('off')
        
        # 预测结果
        pred_colored = create_segmentation_color_map(predictions[i], config.class_colors)
        axes[row, col_base + 2].imshow(pred_colored)
        if i == 0:
            axes[row, col_base + 2].set_title('prediction', fontsize=10)
        axes[row, col_base + 2].axis('off')
    
    # 隐藏多余的子图
    for i in range(n_samples, grid_size[0] * grid_size[1]):
        row = i // grid_size[1]
        col_base = (i % grid_size[1]) * 3
        for j in range(3):
            axes[row, col_base + j].axis('off')
    
    plt.suptitle('comparision_gird', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存对比网格: {save_path}")

# ==================== 主函数 ====================
def main():
    # 配置
    config = VisualizationConfig()
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 参数设置（根据你的实际路径修改）
    best_model_path = "./xxx/best_model.pth"  # 最佳模型路径
    config_path = "./xxx/experiment_20251217_220019_config.json"  # 实验配置
    
    # 数据路径（根据你的实际路径修改）
    DATA_FOLDER = "/root/autodl-tmp/dataset/cropped_Image/{}.jpg"
    DSM_FOLDER = "/root/autodl-tmp/dataset/cropped_Depth/{}.png"
    LABEL_FOLDER = "/root/autodl-tmp/dataset/cropped_Label/{}_label.png"
    
    # 加载测试ID（根据你的实际情况修改）
    def read_txt_to_list(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            return [line.strip() for line in lines if line.strip()]
        except Exception as e:
            print(f"读取文件时出错: {str(e)}")
            return []
    
    test_ids = read_txt_to_list("test1.txt")[:config.num_samples_to_visualize]
    
    if not test_ids:
        print("❌ 没有找到测试ID，请检查test1.txt文件")
        return
    
    print(f"🔍 找到 {len(test_ids)} 个测试样本")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    dataset = VisualizationDataset(test_ids, DATA_FOLDER, DSM_FOLDER, LABEL_FOLDER, transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    # 加载模型
    model, experiment_config = load_best_model(config_path, best_model_path, config.device)
    
    # 进行预测
    print("🎯 开始预测...")
    all_predictions = []
    all_enhanced_imgs = []
    all_samples = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="处理批次"):
            images = batch['image'].to(config.device)
            depths = batch['depth'].to(config.device)
            
            # 预测
            enhanced_imgs, seg_outputs = model(images, depths)
            _, predictions = torch.max(seg_outputs, 1)
            
            # 转移到CPU
            enhanced_imgs = enhanced_imgs.cpu()
            predictions = predictions.cpu()
            
            # 收集结果
            for i in range(len(images)):
                all_samples.append({
                    'id': batch['id'][i],
                    'image': batch['image'][i],
                    'depth': batch['depth'][i],
                    'label': batch['label'][i]
                })
                all_predictions.append(predictions[i].numpy())
                all_enhanced_imgs.append(enhanced_imgs[i])
    
    print(f"✅ 预测完成，共处理 {len(all_samples)} 个样本")
    
    # 可视化结果
    print("🖼️  生成可视化结果...")
    
    # 1. 保存单个样本可视化
    if config.save_individual_images:
        individual_dir = os.path.join(config.output_dir, "individual_samples")
        os.makedirs(individual_dir, exist_ok=True)
        
        for i, sample in enumerate(all_samples):
            save_path = os.path.join(individual_dir, f"sample_{sample['id']}.png")
            visualize_single_sample(sample, all_predictions[i], all_enhanced_imgs[i], 
                                  config, save_path)
    
    # 2. 创建对比网格
    if config.create_comparison_grid:
        grid_path = os.path.join(config.output_dir, "comparison_grid.png")
        create_comparison_grid(all_samples, all_predictions, all_enhanced_imgs, 
                             config, grid_path)
    
    # 3. 创建类别分布图
    all_true_labels = np.concatenate([sample['label'].numpy().ravel() for sample in all_samples])
    all_pred_labels = np.concatenate([pred.ravel() for pred in all_predictions])
    
    distribution_path = os.path.join(config.output_dir, "class_distribution.png")
    create_class_distribution_chart(all_pred_labels, all_true_labels, config, distribution_path)
    
    # 4. 计算并保存指标
    from sklearn.metrics import confusion_matrix, accuracy_score
    
    # 过滤无效标签（255通常表示忽略）
    valid_mask = all_true_labels != 255
    true_valid = all_true_labels[valid_mask]
    pred_valid = all_pred_labels[valid_mask]
    
    # 计算指标
    accuracy = accuracy_score(true_valid, pred_valid)
    cm = confusion_matrix(true_valid, pred_valid, labels=range(16))
    
    # 每类准确率
    class_accuracy = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    mIoU = np.nanmean(class_accuracy)
    
    # 保存指标报告
    report = {
        "总体准确率": float(accuracy),
        "平均交并比(mIoU)": float(mIoU),
        "每类准确率": {config.class_names[i]: float(acc) for i, acc in enumerate(class_accuracy)},
        "处理的样本数量": len(all_samples),
        "模型信息": experiment_config.get('experiment_id', '未知')
    }
    
    with open(os.path.join(config.output_dir, "metrics_report.json"), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "="*50)
    print("📊 可视化结果摘要")
    print("="*50)
    print(f"📁 输出目录: {config.output_dir}")
    print(f"📈 总体准确率: {accuracy:.4f}")
    print(f"🎯 平均交并比(mIoU): {mIoU:.4f}")
    print(f"🖼️  可视化样本数: {len(all_samples)}")
    print("✅ 可视化完成！")
    
    # 生成HTML报告
    generate_html_report(config.output_dir, report, len(all_samples))

def generate_html_report(output_dir, report, num_samples):
    """生成HTML格式的报告"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>语义分割可视化报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #f0f0f0; padding: 20px; border-radius: 10px; }}
            .metrics {{ background: #f9f9f9; padding: 20px; border-radius: 10px; margin: 20px 0; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 语义分割可视化报告</h1>
            <p>生成时间: {report.get('模型信息', '未知实验')}</p>
        </div>
        
        <div class="metrics">
            <h2>📊 性能指标</h2>
            <p><strong>总体准确率:</strong> {report['总体准确率']:.4f}</p>
            <p><strong>平均交并比(mIoU):</strong> {report['平均交并比(mIoU)']:.4f}</p>
            <p><strong>处理样本数:</strong> {num_samples}</p>
        </div>
        
        <h2>🖼️ 可视化结果</h2>
        <div class="grid">
            <div>
                <h3>对比网格</h3>
                <img src="comparison_grid.png" alt="对比网格">
            </div>
            <div>
                <h3>类别分布</h3>
                <img src="class_distribution.png" alt="类别分布">
            </div>
        </div>
        
        <h2>📋 详细结果</h2>
        <p>查看 <code>individual_samples</code> 文件夹获取每个样本的详细可视化结果</p>
        <p>查看 <code>metrics_report.json</code> 获取详细的性能指标</p>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "report.html"), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 HTML报告已生成: {os.path.join(output_dir, 'report.html')}")

if __name__ == "__main__":
    main()