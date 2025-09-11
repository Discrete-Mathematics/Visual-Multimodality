import os 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path 
import glob 
from tqdm import tqdm

def convert_multiclass_to_binary(mask, target_classes=None, background_value=0):
    """
    将多类掩码转换为二值掩码

    TEXT
    Args:
        mask: 输入掩码数组
        target_classes: 要保留为前景的类别列表，如果为None则保留所有非背景类
        background_value: 背景值，默认为0

    Returns:
        binary_mask: 二值掩码 (0=背景, 255=前景)
    """
    binary_mask = np.zeros_like(mask, dtype=np.uint8)

    if target_classes is None:
        # 将所有非背景像素设为前景
        binary_mask[mask != background_value] = 255
    else:
        # 只将指定类别设为前景
        for class_id in target_classes:
            binary_mask[mask == class_id] = 255

    return binary_mask

def analyze_mask_classes(mask_dir, pattern="*_label.png", max_files=100):
    """分析掩码中的所有类别"""
    print("=== 分析掩码类别分布 ===")

    mask_files = glob.glob(os.path.join(mask_dir, pattern))[:max_files]
    all_classes = set()
    class_counts = {}

    for mask_path in tqdm(mask_files, desc="分析掩码"):
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                unique_classes = np.unique(mask)
                all_classes.update(unique_classes)
                
                for class_id in unique_classes:
                    class_counts[class_id] = class_counts.get(class_id, 0) + np.sum(mask == class_id)
        except:
            continue

    print(f"发现的所有类别: {sorted(all_classes)}")
    print("类别像素统计:")
    for class_id in sorted(class_counts.keys()):
        percentage = class_counts[class_id] / sum(class_counts.values()) * 100
        print(f"  类别 {class_id}: {class_counts[class_id]:,} 像素 ({percentage:.2f}%)")

    return sorted(all_classes), class_counts

def convert_masks_batch(input_dir, output_dir, target_classes=None,
                       input_pattern="*_label.png", output_suffix="_binary"):
    """批量转换掩码"""

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有掩码文件
    mask_files = glob.glob(os.path.join(input_dir, input_pattern))
    print(f"找到 {len(mask_files)} 个掩码文件")

    if len(mask_files) == 0:
        print("❌ 没有找到掩码文件")
        return

    # 统计信息
    conversion_stats = {
        'total': len(mask_files),
        'success': 0,
        'failed': 0,
        'empty_after_conversion': 0
    }

    for mask_path in tqdm(mask_files, desc="转换掩码"):
        try:
            # 读取原始掩码
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                conversion_stats['failed'] += 1
                continue
            
            # 转换为二值掩码
            binary_mask = convert_multiclass_to_binary(mask, target_classes)
            
            # 检查转换后是否为空
            if np.count_nonzero(binary_mask) == 0:
                conversion_stats['empty_after_conversion'] += 1
                print(f"⚠️  转换后为空掩码: {os.path.basename(mask_path)}")
            
            # 保存转换后的掩码
            filename = Path(mask_path).stem
            output_path = os.path.join(output_dir, f"{filename}{output_suffix}.png")
            cv2.imwrite(output_path, binary_mask)
            
            conversion_stats['success'] += 1
            
        except Exception as e:
            print(f"❌ 转换失败 {mask_path}: {e}")
            conversion_stats['failed'] += 1

    # 打印统计信息
    print(f"\n=== 转换完成 ===")
    print(f"总文件数: {conversion_stats['total']}")
    print(f"成功转换: {conversion_stats['success']}")
    print(f"转换失败: {conversion_stats['failed']}")
    print(f"转换后为空: {conversion_stats['empty_after_conversion']}")

    return conversion_stats

def visualize_conversion(original_mask_path, converted_mask_path=None,
                        target_classes=None, image_path=None):
    """可视化转换效果"""

    # 读取原始掩码
    original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
    if original_mask is None:
        print(f"❌ 无法读取掩码: {original_mask_path}")
        return

    # 转换掩码
    if converted_mask_path and os.path.exists(converted_mask_path):
        binary_mask = cv2.imread(converted_mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        binary_mask = convert_multiclass_to_binary(original_mask, target_classes)

    # 设置中文字体（解决字体警告）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原始掩码
    axes[0, 0].imshow(original_mask, cmap='tab10', vmin=0, vmax=10)
    axes[0, 0].set_title(f'Original Mask\nClasses: {np.unique(original_mask)}')
    axes[0, 0].axis('off')

    # 转换后掩码
    axes[0, 1].imshow(binary_mask, cmap='gray')
    axes[0, 1].set_title(f'Binary Mask\nForeground pixels: {np.count_nonzero(binary_mask)}')
    axes[0, 1].axis('off')

    # 差异对比
    diff = np.zeros_like(original_mask)
    diff[binary_mask > 0] = 1  # 转换后的前景区域
    diff[original_mask == 0] = 2  # 原始背景区域
    axes[0, 2].imshow(diff, cmap='Set1')
    axes[0, 2].set_title('Conversion Comparison')
    axes[0, 2].axis('off')

    # 如果有图像，显示叠加效果
    if image_path and os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 调整尺寸
            if image_rgb.shape[:2] != original_mask.shape:
                image_rgb = cv2.resize(image_rgb, (original_mask.shape[1], original_mask.shape[0]))
            
            # 原始掩码叠加
            colored_mask1 = np.zeros_like(image_rgb)
            for class_id in np.unique(original_mask):
                if class_id != 0:
                    color = plt.cm.tab10(class_id / 10.0)[:3]
                    colored_mask1[original_mask == class_id] = [c * 255 for c in color]
            
            overlay1 = cv2.addWeighted(image_rgb, 0.7, colored_mask1.astype(np.uint8), 0.3, 0)
            axes[1, 0].imshow(overlay1)
            axes[1, 0].set_title('Image + Original Mask')
            axes[1, 0].axis('off')
            
            # 二值掩码叠加
            colored_mask2 = np.zeros_like(image_rgb)
            colored_mask2[binary_mask > 0] = [255, 0, 0]  # 红色
            
            overlay2 = cv2.addWeighted(image_rgb, 0.7, colored_mask2, 0.3, 0)
            axes[1, 1].imshow(overlay2)
            axes[1, 1].set_title('Image + Binary Mask')
            axes[1, 1].axis('off')
            
            # 原始图像
            axes[1, 2].imshow(image_rgb)
            axes[1, 2].set_title('Original Image')
            axes[1, 2].axis('off')
    else:
        # 显示统计信息
        unique_orig, counts_orig = np.unique(original_mask, return_counts=True)
        axes[1, 0].bar(unique_orig, counts_orig)
        axes[1, 0].set_title('Original Mask Class Distribution')
        axes[1, 0].set_xlabel('Class ID')
        axes[1, 0].set_ylabel('Pixel Count')
        
        unique_bin, counts_bin = np.unique(binary_mask, return_counts=True)
        axes[1, 1].bar(unique_bin, counts_bin)
        axes[1, 1].set_title('Binary Mask Distribution')
        axes[1, 1].set_xlabel('Value (0=bg, 255=fg)')
        axes[1, 1].set_ylabel('Pixel Count')
        
        axes[1, 2].axis('off')

    plt.suptitle(f'Mask Conversion Analysis: {os.path.basename(original_mask_path)}', fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""

    # 设置路径
    input_dir = r"D:\MUSeg\05-Mine\Label"  # 你的掩码文件夹
    output_dir = r"D:\MUSeg\05-Mine\Label_Binary"  # 输出文件夹
    image_dir = r"D:\MUSeg\05-Mine\Image"  # 对应的图像文件夹（可选）

    print("=== SAM 掩码转换工具 ===")

    # 1. 分析现有掩码的类别分布
    print("\n步骤1: 分析掩码类别...")
    all_classes, class_counts = analyze_mask_classes(input_dir, "*_label.png", max_files=50)

    # 2. 根据你的需求选择转换策略
    print(f"\n检测到的类别: {all_classes}")
    print("请选择转换策略:")
    print("1. 将所有非0类别转为前景 (推荐)")
    print("2. 只将特定类别转为前景")
    print("3. 先可视化几个样本再决定")

    choice = input("请输入选择 (1/2/3): ").strip()

    if choice == "3":
        # 可视化几个样本
        sample_mask = os.path.join(input_dir, "05-01-01-0002-231211110630-02-99_label.png")
        sample_image = os.path.join(image_dir, "05-01-01-0003-231211110870-02-99.jpg") if os.path.exists(image_dir) else None
        
        print("可视化样本...")
        visualize_conversion(sample_mask, image_path=sample_image)
        return

    elif choice == "2":
        print("请输入要保留为前景的类别ID（用逗号分隔）:")
        class_input = input("例如: 2,7 表示只保留类别2和7: ").strip()
        try:
            target_classes = [int(x.strip()) for x in class_input.split(',')]
            print(f"将类别 {target_classes} 转换为前景")
        except:
            print("输入格式错误，使用默认策略（所有非0类别）")
            target_classes = None
    else:
        target_classes = None  # 保留所有非0类别
        print("将所有非0类别转换为前景")

    # 3. 执行批量转换
    print(f"\n步骤2: 开始批量转换...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    stats = convert_masks_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        target_classes=target_classes,
        input_pattern="*_label.png"
    )

    # 4. 可视化转换结果
    if stats['success'] > 0:
        print(f"\n步骤3: 可视化转换结果...")
        sample_files = glob.glob(os.path.join(output_dir, "*_binary.png"))[:3]
        for binary_file in sample_files:
            original_name = Path(binary_file).name.replace("_binary.png", "_label.png")
            original_file = os.path.join(input_dir, original_name)
            
            if os.path.exists(original_file):
                image_name = original_name.replace("_label.png", ".jpg")
                image_file = os.path.join(image_dir, image_name) if os.path.exists(image_dir) else None
                
                visualize_conversion(original_file, binary_file, image_path=image_file)

    print(f"\n✅ 转换完成！请将训练脚本中的掩码路径指向: {output_dir}")
    print("建议修改训练脚本中的掩码文件名匹配模式，添加 '_binary' 后缀")

if __name__ == "__main__":
    main()