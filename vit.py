import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from labml_helpers.module import Module
from labml_nn.transformers import TransformerLayer, TransformerConfigs
from labml_nn.utils import clone_module_list
import torch.onnx
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F
import math
import time  # 导入 time 模块

# 强制使用CPU
device = torch.device("cpu")
torch.set_num_threads(4)  # 限制线程数以提高稳定性

# 早停机制
class EarlyStopping:
    def __init__(self, patience=3, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_acc):
        if self.best_score is None or val_acc > self.best_score:
            self.best_score = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# 手动定义 FeedForwardConfigs 类
class FeedForwardConfigs:
    def __init__(self):
        self.d_ff = None
    
    def __call__(self, d_model):
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, d_model)
        )
        return self.ffn

class MultiHeadAttentionConfigs:
    def __init__(self):
        self.heads = None

class CustomTransformerConfigs(TransformerConfigs):
    attn: MultiHeadAttentionConfigs  # 关键修复：添加类型注解
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttentionConfigs()

class PatchEmbeddings(Module):
    def __init__(self, d_model: int, patch_size: int, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, d_model//2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model//2, d_model, patch_size, stride=patch_size)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        bs, c, h, w = x.shape
        x = x.permute(2, 3, 0, 1)
        x = x.view(h * w, bs, c)
        return x

class LearnedPositionalEmbeddings(Module):
    
    def __init__(self, d_model: int, max_len: int = 5_000):
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]]
        return x + pe

class ClassificationHead(Module):
    def __init__(self, d_model, n_hidden, n_classes):
        super().__init__()
        self.linear1 = nn.Linear(d_model, n_hidden)
        self.act = nn.GELU()  # Bug fix: assign GELU to self.act
        self.ln = nn.LayerNorm(n_hidden)  # Bug fix: assign LayerNorm to self.ln
        self.dropout = nn.Dropout(0.1)  # 新增Dropout层
        self.linear2 = nn.Linear(n_hidden, n_classes)
    
    def forward(self, x):
        x = self.act(self.linear1(x))  # 使用 GELU 激活函数
        x = self.dropout(x)  # 应用Dropout
        x = self.ln(x)  # 添加层归一化
        x = self.linear2(x)
        return x

class VisionTransformer(Module):
    def __init__(self, transformer_layer: TransformerLayer, n_layers: int,
                 patch_emb: PatchEmbeddings, pos_emb: LearnedPositionalEmbeddings,
                 classification: ClassificationHead):
        super().__init__()
        self.patch_emb = patch_emb
        self.pos_emb = pos_emb
        self.classification = classification
        self.transformer_layers = clone_module_list(transformer_layer, n_layers)
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, transformer_layer.size), requires_grad=True)
        
        self.attention_pool = nn.Sequential(
            nn.Linear(transformer_layer.size, transformer_layer.size // 4),
            nn.Tanh(),
            nn.Linear(transformer_layer.size // 4, 1)
        )
        
        self.ln = nn.LayerNorm([transformer_layer.size])

    def forward(self, x: torch.Tensor):
        x = self.patch_emb(x)
        cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x], dim=0)
        x = self.pos_emb(x)
        
        residual = x
        for layer in self.transformer_layers:
            x = layer(x=x, mask=None)
            x = x + residual  # 残差连接
            residual = x
        
        attn_scores = self.attention_pool(x)
        attn_weights = torch.softmax(attn_scores, dim=0)
        x = (x * attn_weights).sum(dim=0)
        
        x = self.ln(x)
        x = self.classification(x)
        return x

# 配置训练参数 - 减小模型规模以适应CPU
d_model = 210  # 128→192 (提升50%)
n_layers = 6    # 4→6 (增加层数)
n_hidden = 416  # 256→384 (分类头扩容)
patch_size = 8  # 增大补丁大小以减少序列长度
n_classes = 10
epochs = 20      # 减少训练轮次
batch_size = 64  # 减小批大小
learning_rate = 1e-3

# 简化数据预处理
# 调整为更适合CIFAR-10的增强策略
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 边缘填充裁剪
    transforms.RandomHorizontalFlip(p=0.3),  # 降低翻转概率
    transforms.ColorJitter(brightness=0.15, contrast=0.15),  # 减弱颜色扰动
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.08))  # 降低概率和范围
])

# 加载 CIFAR-10 数据集 - 使用更小的子集进行测试
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
train_dataset = torch.utils.data.Subset(train_dataset, range(0, 40000))  # 只使用50000个样本
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
test_dataset = torch.utils.data.Subset(test_dataset, range(0, 1000))  # 只使用1000个测试样本
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 配置 Transformer 层
transformer_configs = CustomTransformerConfigs()
transformer_configs.dropout = 0.1  # 新增残差连接后的Dropout
transformer_configs.d_model = d_model
transformer_configs.ffn = FeedForwardConfigs()
transformer_configs.ffn.d_ff = d_model * 4  # 减小前馈网络维度
transformer_configs.ffn(d_model)
transformer_configs.attn = MultiHeadAttentionConfigs()
transformer_configs.attn.heads = 6  # 减少注意力头数
transformer_configs.n_heads = 6


transformer_layer = transformer_configs.encoder_layer

# 初始化模型
patch_emb = PatchEmbeddings(d_model, patch_size, 3)
pos_emb = LearnedPositionalEmbeddings(d_model)
classification = ClassificationHead(d_model, n_hidden, n_classes)
model = VisionTransformer(transformer_layer, n_layers, patch_emb, pos_emb, classification).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), 
                             lr=5e-4, 
                             weight_decay=0.001) # AdamW 优化器

# 添加学习率预热和余弦退火
from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(current_step):
    warmup_steps = 500
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, epochs*len(train_loader) - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda)

# 训练和验证函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_grad_norm = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        total_grad_norm += grad_norm.item()
        
        optimizer.step()  # 仅调用一次
        scheduler.step()  # 每个batch更新学习率

        running_loss += loss.item()
    avg_grad_norm = total_grad_norm / len(train_loader)
    return running_loss / len(train_loader), avg_grad_norm

def validate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 初始化早停机制
early_stopping = EarlyStopping(patience=3, verbose=True)

# 训练循环
train_losses = []
val_accuracies = []
grad_norms = []

for epoch in range(epochs):
    start_time = time.time()  # 记录训练开始时间
    train_loss, avg_grad_norm = train(model, train_loader, criterion, optimizer, device)
    val_accuracy = validate(model, test_loader, device)
    end_time = time.time()  # 记录训练结束时间
    epoch_time = end_time - start_time  # 计算本轮训练耗时
    
    train_losses.append(train_loss)
    val_accuracies.append(val_accuracy)
    grad_norms.append(avg_grad_norm)
    
    print(f'Epoch {epoch+1}/{epochs}, '
          f'Train Loss: {train_loss:.4f}, '
          f'Val Acc: {val_accuracy:.2f}%, '
          f'Grad Norm: {avg_grad_norm:.2f}, '
          f'Time: {epoch_time:.2f}s')  # 打印训练耗时
    
    # 使用早停机制
    early_stopping(val_accuracy)
    if early_stopping.early_stop:
        print("Early stopping")
        break

print('Training finished.')

# 保存模型为 PyTorch 格式
pth_path = "trained_vit_model.pth"
torch.save(model.state_dict(), pth_path)
print(f"Model has been saved to {pth_path}")

# 尝试再次保存为 ONNX 格式
dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
onnx_path = "trained_vit_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"Model has been saved to {onnx_path}")