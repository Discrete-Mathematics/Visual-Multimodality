import torch
from torch import nn,optim
from labml_helpers.module import Module
from labml_nn.transformers import TransformerLayer
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.utils import clone_module_list 
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader 
import math

class MultiHeadAttention(nn.Module):  
    def __init__(self, d_model: int, n_heads: int, dropout_prob: float = 0.1):  
        super().__init__()  
        assert d_model % n_heads == 0  # Ensure d_model is divisible by n_heads  

        self.d_head = d_model // n_heads  
        self.n_heads = n_heads  
        self.linear_q = nn.Linear(d_model, d_model)  
        self.linear_k = nn.Linear(d_model, d_model)  
        self.linear_v = nn.Linear(d_model, d_model)  
        self.dropout = nn.Dropout(dropout_prob)  
        self.linear_out = nn.Linear(d_model, d_model)  

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):  
        batch_size = query.size(0)  

        # Linear projections  
        Q = self.linear_q(query).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)  # (batch, heads, seq_len, d_head)  
        K = self.linear_k(key).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)  
        V = self.linear_v(value).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)  

        # Scaled dot-product attention  
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  
        if mask is not None:  
            scores.masked_fill_(mask == 0, float('-inf'))  
        attn_weights = torch.softmax(scores, dim=-1)  
        attn_weights = self.dropout(attn_weights)  

        # Attention output  
        output = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_head)  

        return self.linear_out(output)  

# 将图像分割成多个patch，并将其展平为序列
class PatchEmbeddings(Module):  
    def __init__(self, d_model: int, patch_size: int, in_channels: int):  
        super().__init__()  
        self.conv = nn.Conv2d(in_channels, d_model, patch_size, stride=patch_size)  

    def forward(self, x: torch.Tensor):  
        x = self.conv(x)  
        bs, c, h, w = x.shape  
        x = x.permute(0, 2, 3, 1)  
        x = x.view(bs, h * w, c)  
        return x  

# 将学习的位置嵌入添加到补丁嵌入中
class LearnedPositionalEmbeddings(Module):
    def __init__(self, d_model: int, max_len: int = 5_000):
        super().__init__() 
        
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]]
        return x + pe

# 分类头
class ClassificationHead(Module):
    def __init__(self, d_model: int, n_hidden: int, n_classes: int):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, n_hidden)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_classes)
    
    def forward(self, x: torch.Tensor):
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        return x

# vit模型
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

        self.ln = nn.LayerNorm(transformer_layer.size)

    def forward(self, x: torch.Tensor):
        x = self.patch_emb(x)
        cls_token_emb = self.cls_token_emb.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token_emb, x], dim=1)
        x = self.pos_emb(x)

        for layer in self.transformer_layers:
            x = layer(x=x, mask=None)
        
        x = x[:, 0]
        x = self.ln(x)
        x = self.classification(x)
        return x

if __name__ == "__main__":
    # Hyperparameters  
    d_model = 768        # Dimension of the model  
    n_hidden = 512       # Hidden dimension in the classification head  
    n_classes = 10       # Number of classes in the dataset (for CIFAR-10)  
    n_layers = 6         # Number of transformer layers  
    patch_size = 16      # Size of patches  
    in_channels = 3      # Number of color channels (RGB)
    dropout_rate = 0.1  
    feed_forward = FeedForward(d_model=d_model, d_ff=2048, activation=nn.ReLU())
    self_attention = MultiHeadAttention(d_model=d_model, n_heads=8, dropout_prob=dropout_rate)

    # Create data transformations  
    transform = transforms.Compose([  
        transforms.Resize((32, 32)),  
        transforms.ToTensor(),  
    ])  

    # Load CIFAR-10 dataset  
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)  
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  

    # Initialize the components  
    patch_emb = PatchEmbeddings(d_model=d_model, patch_size=patch_size, in_channels=in_channels)  
    pos_emb = LearnedPositionalEmbeddings(d_model=d_model, max_len=1024)  
    classification_head = ClassificationHead(d_model=d_model, n_hidden=n_hidden, n_classes=n_classes)  
    transformer_layer = TransformerLayer(d_model=d_model, dropout_prob=dropout_rate, self_attn=self_attention, feed_forward=feed_forward)  # Assuming TransformerLayer is defined  
    model = VisionTransformer(transformer_layer, n_layers, patch_emb, pos_emb, classification_head)  

    # Define loss function and optimizer  
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  

    # Training loop  
    model.train()  
    for epoch in range(10):  # Train for 10 epochs  
        for images, labels in train_loader:  
            optimizer.zero_grad()  
            outputs = model(images)  # Forward pass  
            loss = criterion(outputs, labels)  # Compute loss  
            loss.backward()  # Backward pass  
            optimizer.step()  # Update weights  
            
        print(f'Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}')  

    torch.save(model.state_dict(), 'vision_transformer_cifar10.pth')  
    print("Model saved as 'vision_transformer_cifar10.pth'")  

    # Evaluation on the test set  
    model.eval()  # Set the model to evaluation mode  
    correct = 0  
    total = 0  

    with torch.no_grad():  # Disable gradient calculation for evaluation  
        for images, labels in train_loader:  
            outputs = model(images)  # Forward pass  
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class  
            total += labels.size(0)  # Update total number of images  
            correct += (predicted == labels).sum().item()  # Update correct predictions  

    accuracy = 100 * correct / total  
    print(f'Accuracy on the CIFAR-10 test set: {accuracy:.2f}%')  
