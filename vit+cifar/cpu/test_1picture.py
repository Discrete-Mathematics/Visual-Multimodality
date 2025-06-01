import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import onnxruntime as ort  # 导入 onnxruntime

# 修改 Resize 操作，明确指定图像尺寸为 32x32
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 明确指定高度和宽度
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# CIFAR-10类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# 加载 ONNX 模型
onnx_path = "trained_vit_model.onnx"
ort_session = ort.InferenceSession(onnx_path)

def predict_and_visualize(image_path, ort_session, transform):
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    
    # 预处理并添加batch维度
    input_tensor = transform(image).unsqueeze(0)  # [1, 3, 32, 32]
    input_tensor = input_tensor.numpy()  # 转换为 numpy 数组

    # 打印输入张量的形状，用于调试
    print(f"Input tensor shape: {input_tensor.shape}")

    # 预测
    input_name = ort_session.get_inputs()[0].name
    with torch.no_grad():
        output = ort_session.run(None, {input_name: input_tensor})[0]
        output = torch.tensor(output)  # 转换为 torch 张量
        probs = torch.softmax(output, dim=1)[0]
        pred_class = classes[torch.argmax(probs)]
    
    # 可视化
    plt.figure(figsize=(12, 4))
    
    # 原始图片
    plt.subplot(1, 3, 1)
    plt.imshow(np.array(image.resize((128, 128))))  # 放大显示
    plt.title(f"Input Image\nSize: {image.size}")
    plt.axis('off')
    
    # 预处理后的张量
    plt.subplot(1, 3, 2)
    denorm = transforms.Normalize(
        mean=[-0.4914/0.247, -0.4822/0.2435, -0.4465/0.2616],
        std=[1/0.247, 1/0.2435, 1/0.2616]
    )
    prep_img = denorm(torch.tensor(input_tensor[0])).permute(1, 2, 0).clamp(0, 1)
    plt.imshow(prep_img)
    plt.title("Preprocessed Tensor")
    plt.axis('off')
    
    # 预测结果
    plt.subplot(1, 3, 3)
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    plt.barh(classes, probs.numpy(), color=colors)
    plt.xlabel("Probability")
    plt.xlim(0, 1)
    plt.title(f"Prediction: {pred_class}\nConfidence: {probs.max():.1%}")
    
    plt.tight_layout()
    plt.show()
    
    return pred_class, probs

if __name__ == "__main__":
    image_path = r"D:\fuwuqi\vit+cifar训练\cpu\R-C.jpg"  # 替换为实际图片路径
    pred_class, probs = predict_and_visualize(image_path, ort_session, test_transform)
    print(f"Predicted class: {pred_class}")