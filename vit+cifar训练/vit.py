import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
# from labml_nn.transformers.feed_forward import FeedForwardConfigs
from labml_helpers.module import Module
from labml_nn.transformers import TransformerLayer
#from labml_nn.transformers.mha import MultiHeadAttentionConfigs
from labml_nn.transformers import TransformerConfigs
from labml_nn.utils import clone_module_list

# 手动定义 FeedForwardConfigs 类
class FeedForwardConfigs:
    def __init__(self):
        # 前馈网络的隐藏层维度
        self.d_ff = None
    def __call__(self, d_model):
        # 初始化前馈网络层
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, d_model)
        )
        return self.ffn

class MultiHeadAttentionConfigs:
    def __init__(self):
        # 多头注意力的头数
        self.heads = None
# 手动扩展 TransformerConfigs 类
class CustomTransformerConfigs(TransformerConfigs):
    attn: MultiHeadAttentionConfigs  # 为 attn 属性添加类型注解
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttentionConfigs()

class PatchEmbeddings(Module):
    def __init__(self, d_model: int, patch_size: int, in_channels: int):
        """
        初始化补丁嵌入层。

        :param d_model: 变压器嵌入的大小。
        :param patch_size: 补丁的大小。
        :param in_channels: 输入图像中的通道数（RGB 为 3）。
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, d_model, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        """
        前向传播方法。

        :param x: 形状为 [batch_size, channels, height, width] 的输入图像。
        :return: 形状为 [patches, batch_size, d_model] 的补丁嵌入。
        """
        x = self.conv(x)
        bs, c, h, w = x.shape
        x = x.permute(2, 3, 0, 1)
        x = x.view(h * w, bs, c)
        return x

class LearnedPositionalEmbeddings(Module):
    def __init__(self, d_model: int, max_len: int = 5_000):
        """
        初始化学习位置嵌入层。

        :param d_model: 变压器嵌入的大小。
        :param max_len: 补丁的最大数量。
        """
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
        前向传播方法。

        :param x: 形状为 [patches, batch_size, d_model] 的补丁嵌入。
        :return: 添加位置嵌入后的补丁嵌入。
        """
        pe = self.positional_encodings[:x.shape[0]]
        return x + pe

class ClassificationHead(Module):
    def __init__(self, d_model: int, n_hidden: int, n_classes: int):
        """
        初始化分类头。

        :param d_model: 变压器嵌入的大小。
        :param n_hidden: 隐藏层的大小。
        :param n_classes: 分类任务中的类数。
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, n_hidden)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x: torch.Tensor):
        """
        前向传播方法。

        :param x: [CLS] 令牌的转换器编码。
        :return: 分类的 logits。
        """
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        return x

class VisionTransformer(Module):
    def __init__(self, transformer_layer: TransformerLayer, n_layers: int,
                 patch_emb: PatchEmbeddings, pos_emb: LearnedPositionalEmbeddings,
                 classification: ClassificationHead):
        """
        初始化视觉变压器模型。

        :param transformer_layer: 单个变压器层的副本。
        :param n_layers: 变压器层的数量。
        :param patch_emb: 补丁嵌入层。
        :param pos_emb: 位置嵌入层。
        :param classification: 分类头。
        """
        super().__init__()
        self.patch_emb = patch_emb
        self.pos_emb = pos_emb
        self.classification = classification
        self.transformer_layers = clone_module_list(transformer_layer, n_layers)
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, transformer_layer.size), requires_grad=True)
        self.ln = nn.LayerNorm([transformer_layer.size])

    def forward(self, x: torch.Tensor):
        """
        前向传播方法。

        :param x: 形状为 [batch_size, channels, height, width] 的输入图像。
        :return: 分类的 logits。
        """
        x = self.patch_emb(x)
        cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x])
        x = self.pos_emb(x)
        for layer in self.transformer_layers:
            x = layer(x=x, mask=None)
        x = x[0]
        x = self.ln(x)
        x = self.classification(x)
        return x

# 配置训练参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_model = 128
patch_size = 4
n_layers = 4
n_hidden = 128
n_classes = 10
epochs = 20
batch_size = 128
learning_rate = 1e-3

# 数据预处理
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 配置 Transformer 层
transformer_configs = CustomTransformerConfigs()
transformer_configs.d_model = d_model
transformer_configs.ffn = FeedForwardConfigs()
transformer_configs.ffn.d_ff = d_model * 4
transformer_configs.ffn(d_model)
transformer_configs.attn = MultiHeadAttentionConfigs()
transformer_configs.attn.heads = 8
transformer_configs.n_heads = 8

transformer_layer = transformer_configs.encoder_layer

# 初始化模型
patch_emb = PatchEmbeddings(d_model, patch_size, 3)
pos_emb = LearnedPositionalEmbeddings(d_model)
classification = ClassificationHead(d_model, n_hidden, n_classes)
model = VisionTransformer(transformer_layer, n_layers, patch_emb, pos_emb, classification).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练和验证函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

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

# 训练循环
for epoch in range(epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_accuracy = validate(model, test_loader, device)
    scheduler.step()

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

print('Training finished.')