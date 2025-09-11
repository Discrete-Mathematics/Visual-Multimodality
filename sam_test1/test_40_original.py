import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForMaskGeneration
from torchmetrics import JaccardIndex
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def check_paths_and_files(image_dir, label_dir):
    """检查路径和文件是否存在"""
    print("=== 路径检查 ===")
    print(f"图像目录: {image_dir} - 存在: {os.path.exists(image_dir)}")
    print(f"标签目录: {label_dir} - 存在: {os.path.exists(label_dir)}")

    if os.path.exists(image_dir):
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"找到 {len(image_files)} 张图像")
        if image_files:
            print(f"前5张图像: {image_files[:5]}")

    if os.path.exists(label_dir):
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.png') and '_label_binary' in f]
        print(f"找到 {len(label_files)} 个标签文件")
        if label_files:
            print(f"前5个标签: {label_files[:5]}")

def enhance_image(image):
    """增强低光照图像"""
    img_np = np.array(image)

    # 应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # 转换到LAB颜色空间
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # 对L通道应用CLAHE
    l_clahe = clahe.apply(l)

    # 合并通道并转换回RGB
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced)

def get_strategic_points(mask_np, num_points=5):
    """改进的点选择策略"""
    binary_mask = mask_np.astype(bool)

    if not np.any(binary_mask):
        return None, None

    # 计算质心
    y_indices, x_indices = np.where(binary_mask)
    centroid_x = int(np.mean(x_indices))
    centroid_y = int(np.mean(y_indices))

    # 从质心向外选择点
    points = []
    labels = []

    # 添加质心
    points.append([centroid_x, centroid_y])
    labels.append(1)

    # 添加边界点
    if len(x_indices) > 0:
        # 最左、最右、最上、最下点
        points.append([int(np.min(x_indices)), centroid_y])
        points.append([int(np.max(x_indices)), centroid_y])
        points.append([centroid_x, int(np.min(y_indices))])
        points.append([centroid_x, int(np.max(y_indices))])
        labels.extend([1, 1, 1, 1])

    # 确保不超过请求的点数
    points = points[:num_points]
    labels = labels[:num_points]

    return points, labels

def get_negative_points(mask_np, num_points=3, positive_points=None):
    """改进的负点选择策略"""
    background_mask = ~mask_np.astype(bool)

    if not np.any(background_mask):
        return None, None

    height, width = mask_np.shape
    points = []

    # 在真实背景区域中选择负点
    y_indices, x_indices = np.where(background_mask)

    if len(y_indices) == 0:
        return None, None

    # 如果有正点，选择远离正点的负点
    if positive_points and len(positive_points) > 0:
        # 计算正点的平均位置
        avg_x = np.mean([p[0] for p in positive_points])
        avg_y = np.mean([p[1] for p in positive_points])

        # 计算每个背景点到正点中心的距离
        distances = []
        for i in range(len(y_indices)):
            dist = np.sqrt((x_indices[i] - avg_x)**2 + (y_indices[i] - avg_y)**2)
            distances.append((x_indices[i], y_indices[i], dist))

        # 选择距离最远的点作为负点
        distances.sort(key=lambda x: x[2], reverse=True)
        for i in range(min(num_points, len(distances))):
            points.append([int(distances[i][0]), int(distances[i][1])])
    else:
        # 随机选择背景点
        indices = np.random.choice(len(y_indices), min(num_points, len(y_indices)), replace=False)
        for i in indices:
            points.append([int(x_indices[i]), int(y_indices[i])])

    labels = [0] * len(points)
    return points, labels

def get_bounding_box(mask_np, padding_ratio=0.1):
    """改进的边界框生成 - 添加适当的padding"""
    binary_mask = mask_np.astype(bool)

    if not np.any(binary_mask):
        return None

    # 找到非零像素的坐标
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # 基于目标大小的比例添加padding
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    padding_x = int(bbox_width * padding_ratio)
    padding_y = int(bbox_height * padding_ratio)

    xmin = max(0, xmin - padding_x)
    ymin = max(0, ymin - padding_y)
    xmax = min(mask_np.shape[1] - 1, xmax + padding_x)
    ymax = min(mask_np.shape[0] - 1, ymax + padding_y)

    # SAM需要特定的边界框格式：[[[xmin, ymin, xmax, ymax]]]
    return [[[int(xmin), int(ymin), int(xmax), int(ymax)]]]

def refine_mask_with_morphology(mask):
    """使用形态学操作细化掩码"""
    # 转换为uint8
    mask_uint8 = mask.astype(np.uint8) * 255

    # 先进行闭操作填充小洞
    kernel = np.ones((3, 3), np.uint8)
    refined = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    # 再进行开操作去除小噪声
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

    return refined > 0

def post_process_mask(pred_mask, min_size_ratio=0.3):
    """后处理掩码 - 移除小区域并选择最大连通区域"""
    # 形态学处理
    pred_mask_refined = refine_mask_with_morphology(pred_mask)

    # 找到连通区域
    labeled_pred, num_features = ndimage.label(pred_mask_refined)

    if num_features == 0:
        return pred_mask_refined

    # 计算每个区域的大小
    sizes = ndimage.sum(pred_mask_refined, labeled_pred, range(1, num_features + 1))

    if len(sizes) == 0:
        return pred_mask_refined

    # 找到最大区域
    max_size = np.max(sizes)
    min_size = max_size * min_size_ratio

    # 创建新的掩码，只保留足够大的区域
    final_mask = np.zeros_like(pred_mask_refined)
    for i in range(1, num_features + 1):
        if sizes[i-1] >= min_size:
            final_mask[labeled_pred == i] = True

    return final_mask

def visualize_comparison(image, true_mask, pred_mask, points, image_name, save_dir="debug_vis"):
    """可视化比较结果"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原始图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 真实掩码
    axes[0, 1].imshow(true_mask, cmap='gray')
    axes[0, 1].set_title('True Mask')
    axes[0, 1].axis('off')

    # 预测掩码
    axes[0, 2].imshow(pred_mask, cmap='gray')
    axes[0, 2].set_title('Predicted Mask')
    axes[0, 2].axis('off')

    # 叠加真实掩码
    axes[1, 0].imshow(image)
    axes[1, 0].imshow(true_mask, alpha=0.5, cmap='Reds')
    axes[1, 0].set_title('Image + True Mask')
    axes[1, 0].axis('off')

    # 叠加预测掩码
    axes[1, 1].imshow(image)
    axes[1, 1].imshow(pred_mask, alpha=0.5, cmap='Blues')
    axes[1, 1].set_title('Image + Predicted Mask')
    axes[1, 1].axis('off')

    # 点和掩码
    axes[1, 2].imshow(image)
    if points:
        # 分离正点和负点
        pos_points = [p for i, p in enumerate(points) if i < len(points) - (len(points) - len([p for p in points if isinstance(p, list)]))]
        neg_points = [p for i, p in enumerate(points) if i >= len(pos_points)]

        if pos_points:
            axes[1, 2].scatter([p[0] for p in pos_points], [p[1] for p in pos_points],
                              c='red', s=50, marker='x', label='Positive')
        if neg_points:
            axes[1, 2].scatter([p[0] for p in neg_points], [p[1] for p in neg_points],
                              c='blue', s=50, marker='o', label='Negative')

    axes[1, 2].imshow(pred_mask, alpha=0.3, cmap='Blues')
    axes[1, 2].set_title('Points + Prediction')
    axes[1, 2].axis('off')
    if points:
        axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{image_name}_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # 设置目录 - 与训练代码保持一致
    image_dir = "/root/01-Mine/Image"
    label_dir = "/root/01-Mine/Label_Binary"

    # 检查路径
    check_paths_and_files(image_dir, label_dir)

    # 加载模型和处理器
    print("\n=== 加载模型 ===")
    model_id = "facebook/sam-vit-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMaskGeneration.from_pretrained(model_id)

    checkpoint_path = "/root/original_train.pth"
    print(f"加载检查点: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点文件 {checkpoint_path} 不存在!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 确保模型结构与检查点匹配
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("模型权重加载成功!")
    except Exception as e:
        print(f"模型加载错误: {e}")
        print("尝试仅加载mask decoder...")
        model.mask_decoder.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()
    print(f"模型已移动到: {device}")

    # 初始化 IoU 计算器
    jaccard = JaccardIndex(task="binary", num_classes=2).to(device)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 只测试前39张图像
    image_files = image_files[:39]
    
    iou_scores = []

    print(f"\n=== 开始处理前 {len(image_files)} 张图像 ===")

    # 遍历图像
    for image_file in image_files:
        try:
            base_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, base_name + '_label_binary.png')

            # 检查标签文件是否存在
            if not os.path.exists(label_path):
                print(f"跳过 {image_file}: 标签文件不存在")
                continue

            # 加载图像和掩码
            original_image = Image.open(image_path).convert("RGB")
            true_mask_img = Image.open(label_path)
            true_mask_np = np.array(true_mask_img)

            # 使用与训练时一致的二值化方式
            if true_mask_np.max() > 1:
                true_mask_binary = (true_mask_np > 127)
            else:
                true_mask_binary = (true_mask_np > 0.5)

            print(f"\n处理 {image_file}: 掩码形状 {true_mask_np.shape}, 唯一值 {np.unique(true_mask_np)}")
            print(f"二值掩码中目标像素数量: {np.sum(true_mask_binary)}")

            if not np.any(true_mask_binary):
                print(f"图像 {image_file}: 没有找到目标像素，跳过...")
                continue

            # 使用原始图像或增强图像
            image = original_image  # 暂时禁用增强

            # 获取策略性点提示
            points, labels = get_strategic_points(true_mask_binary, num_points=3)
            if points is None:
                print(f"图像 {image_file}: 无法找到有效点")
                continue

            # 获取负点提示
            neg_points, neg_labels = get_negative_points(true_mask_binary, num_points=2, positive_points=points)

            # 获取边界框提示
            bbox = get_bounding_box(true_mask_binary, padding_ratio=0.05)

            # 合并点提示
            all_points = points
            all_labels = labels

            if neg_points:
                all_points.extend(neg_points)
                all_labels.extend(neg_labels)

            # 确保所有点都在图像范围内
            height, width = true_mask_np.shape
            valid_points = []
            valid_labels = []
            for point, label in zip(all_points, all_labels):
                x, y = point
                if 0 <= x < width and 0 <= y < height:
                    valid_points.append(point)
                    valid_labels.append(label)

            if not valid_points:
                print(f"图像 {image_file}: 过滤后没有有效点")
                continue

            input_points = [valid_points]
            input_labels = [valid_labels]

            print(f"正点: {points}")
            if neg_points:
                print(f"负点: {neg_points}")
            print(f"边界框: {bbox}")

            # 准备输入
            inputs = processor(
                image,
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=bbox if bbox else None,
                return_tensors="pt"
            ).to(device)

            # 模型预测
            with torch.no_grad():
                outputs = model(**inputs)

            # 后处理
            predicted_masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
                mask_threshold=0.0,
                binarize=False
            )

            # 选择最佳掩码
            scores = outputs.iou_scores.cpu()
            best_mask_idx = torch.argmax(scores[0, 0]).item()

            print(f"预测掩码数量: {len(predicted_masks[0][0])}")
            print(f"IoU 分数: {scores[0, 0].tolist()}")

            pred_mask = predicted_masks[0][0][best_mask_idx].numpy()
            pred_mask_binary = pred_mask > 0.5

            # 后处理
            pred_mask_processed = post_process_mask(pred_mask_binary, min_size_ratio=0.2)

            # 调整预测掩码大小
            pred_mask_resized = np.array(Image.fromarray(pred_mask_processed.astype(np.uint8)).resize(
                true_mask_img.size, Image.NEAREST)).astype(bool)

            # 计算IoU
            pred_tensor = torch.from_numpy(pred_mask_resized.astype(np.int32)).to(device)
            true_tensor = torch.from_numpy(true_mask_binary.astype(np.int32)).to(device)

            iou = jaccard(pred_tensor, true_tensor)
            iou_scores.append(iou.item())

            print(f"最终 IoU: {iou.item():.4f}")

            # 可视化比较
            if iou.item() < 0.8:  # 只保存效果不好的可视化
                visualize_comparison(
                    original_image, true_mask_binary, pred_mask_resized,
                    valid_points, base_name
                )
                print(f"可视化已保存: {base_name}")

            print("-" * 50)

        except Exception as e:
            print(f"处理 {image_file} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 计算结果
    if iou_scores:
        miou = np.mean(iou_scores)
        print(f"\n=== 最终结果 ===")
        print(f"平均 IoU: {miou:.4f}")
        print(f"成功处理 {len(iou_scores)} 张图像")
        print(f"IoU 范围: {min(iou_scores):.4f} - {max(iou_scores):.4f}")
        print(f"IoU 标准差: {np.std(iou_scores):.4f}")
    else:
        print("没有成功处理任何图像")

if __name__ == "__main__":
    main()