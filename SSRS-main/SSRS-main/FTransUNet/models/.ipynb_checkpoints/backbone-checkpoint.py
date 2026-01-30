import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck  # 移除 model_urls 导入

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=None, pretrained=False, in_channels=3):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.inplanes = 64

        # 输入卷积层（支持自定义输入通道数）
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 骨干层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 分类层（可选）
        self.fc = None
        if num_classes is not None and num_classes > 0:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 加载预训练权重
        if pretrained:
            self._load_pretrained_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
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
        适配新版torchvision：不再依赖 model_urls，直接加载官方ResNet50预训练权重
        两种方式二选一（以下实现方式1，更简洁）
        """
        # 方式1：直接使用torchvision内置的ResNet50加载预训练权重（推荐，简洁高效）
        pretrained_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # 新版torchvision指定权重方式
        
        # 方式2：手动指定官方权重URL（备用，若方式1报错可使用）
        # import os
        # from torch.hub import load_state_dict_from_url
        # RESNET50_URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
        # pretrained_dict = load_state_dict_from_url(RESNET50_URL, progress=True)
        # pretrained_resnet50 = models.resnet50()
        # pretrained_resnet50.load_state_dict(pretrained_dict)

        # 获取当前模型的权重字典
        model_dict = self.state_dict()

        # 过滤掉不匹配的权重（主要是fc层、输入卷积层<若in_channels≠3>）
        pretrained_dict = pretrained_resnet50.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

        # 更新当前模型权重
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print("成功加载ImageNet预训练权重（骨干层）")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.num_classes is not None and self.num_classes > 0:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def ResNet50(num_classes=None, pretrained=False, in_channels=3):
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels
    )