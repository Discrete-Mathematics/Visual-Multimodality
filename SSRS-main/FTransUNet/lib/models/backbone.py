import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck, BasicBlock, model_urls

class ResNet(nn.Module):
    """
    通用ResNet骨干网络（支持ResNet18/34/50/101/152）
    适配特征提取场景，可去除分类层，输出指定通道数的特征图
    """
    def __init__(self, block, layers, num_classes=None, pretrained=False):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.inplanes = 64

        # 1. 初始卷积层（输入层）：3通道→64通道，下采样2倍
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2. ResNet骨干层（layer1-layer4）：核心特征提取
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 3. 分类层（仅当num_classes不为None时启用）
        self.fc = None
        if num_classes is not None and num_classes > 0:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 4. 加载预训练权重（若指定）
        if pretrained:
            self._load_pretrained_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建ResNet的残差块层（layer1-layer4）
        block: 残差块类型（Bottleneck for ResNet50+/BasicBlock for ResNet18/34）
        planes: 输出通道数
        blocks: 该层包含的残差块数量
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _load_pretrained_weights(self):
        """
        加载ImageNet预训练权重（适配torchvision官方权重）
        自动忽略分类层（fc）的权重不匹配（当num_classes=None时）
        """
        # 加载官方ResNet50预训练权重
        pretrained_dict = models.utils.load_state_dict_from_url(
            model_urls['resnet50'],
            progress=True
        )

        # 获取当前模型的权重字典
        model_dict = self.state_dict()

        # 过滤掉不匹配的权重（主要是fc层，当num_classes=None时）
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # 更新当前模型权重
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print("成功加载ImageNet预训练权重（仅骨干层，忽略不匹配的分类层权重）")

    def forward(self, x):
        """
        前向传播：
        - 当num_classes=None时：输出layer4的特征图（2048通道，尺寸为输入1/32）
        - 当num_classes≠None时：输出分类结果（与标准ResNet一致）
        """
        # 初始卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 骨干层特征提取
        x = self.layer1(x)  # 64通道，输入1/4
        x = self.layer2(x)  # 128通道，输入1/8
        x = self.layer3(x)  # 256通道，输入1/16
        x = self.layer4(x)  # 512*4=2048通道，输入1/32（Bottleneck.expansion=4）

        # 分类层（可选）
        if self.num_classes is not None and self.num_classes > 0:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        # 返回结果：特征图（分类层关闭）或分类结果（分类层开启）
        return x

def ResNet50(num_classes=None, pretrained=False):
    """
    封装ResNet50，适配阶段2初始化代码的快捷接口
    对应参数：
    - num_classes: None（默认，仅特征提取）/ 分类类别数
    - pretrained: True（加载ImageNet预训练）/ False（随机初始化）
    """
    # ResNet50使用Bottleneck残差块，layers=[3,4,6,3]为标准ResNet50配置
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        pretrained=pretrained
    )