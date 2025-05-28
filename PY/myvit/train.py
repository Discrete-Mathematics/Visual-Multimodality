import os, random, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from vit import create_vit


def train_epoch(model, data_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(data_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device) # 将数据移动到GPU/CPU上
        
        optimizer.zero_grad()
        outputs = model(images)   # 前向传播, 将输入图像传递给模型，得到模型的输出。
        loss = criterion(outputs, labels) # 损失函数, 计算模型输出与真实标签之间的损失
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(data_loader), 100. * correct / total


def validate(model, data_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(data_loader), 100. * correct / total

def read_split_data(root: str, val_rate: float = 0.2):
    """数据集划分函数"""
    random.seed(0)  # 保证随机结果可复现

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()  # 排序，保证各平台顺序一致
    
    # 生成类别名称以及对应的数字索引
    class_indices = {k: v for v, k in enumerate(flower_class)}
    
    # 保存类别索引映射
    with open('class_indices.json', 'w') as json_file:
        json.dump({v: k for k, v in class_indices.items()}, json_file, indent=4)

    train_images_path = []      # 存储训练集的所有图片路径
    train_images_label = []     # 存储训练集图片对应索引信息
    val_images_path = []        # 存储验证集的所有图片路径
    val_images_label = []       # 存储验证集图片对应索引信息
        
    # 遍历每个类别文件夹
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 获取该类别下所有文件
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.isfile(os.path.join(cla_path, i))]
        images.sort()  # 排序，保证各平台顺序一致
        image_class = class_indices[cla]  # 获取该类别对应的索引
        val_path = random.sample(images, k=int(len(images) * val_rate))          # 按比例随机采样验证样本

        # 划分训练集和验证集
        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print(f"图片总数: {len(train_images_path) + len(val_images_path)}")
    print(f"训练集: {len(train_images_path)}")
    print(f"测试集: {len(val_images_path)}")

    return train_images_path, train_images_label, val_images_path, val_images_label

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 5  # 5类
    batch_size = 16  
    epochs = 15        
    lr = 1e-4
    data_path = "datasets\\flower_photos"
    
    print(f"Using device: {device}")
    os.makedirs("./weights", exist_ok=True)
    
    # 数据处理
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # 加载数据集
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)
    
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path), transform=transform_train)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_path), transform=transform_val)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = create_vit(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()        # nn中自带的交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练
    best_acc = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # 训练/验证
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"./weights/best.pth")
    
    print(f"\n训练完成, 最佳模型测试精度: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
