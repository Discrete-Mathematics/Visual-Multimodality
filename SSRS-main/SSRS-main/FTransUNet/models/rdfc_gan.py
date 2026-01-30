import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """残差块，用于生成器主干"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x

class GeneratorA2B(nn.Module):
    """G_A2B：残缺深度图 → 补全深度图（深度补全生成器）"""
    def __init__(self, in_channels=1, out_channels=1, ngf=64, n_res_blocks=9):
        super(GeneratorA2B, self).__init__()
        # 下采样
        self.conv1 = nn.Conv2d(in_channels, ngf, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*2)
        self.conv3 = nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*4)

        # 残差块
        self.res_blocks = nn.Sequential(*[ResidualBlock(ngf*4) for _ in range(n_res_blocks)])

        # 上采样
        self.deconv1 = nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf*2)
        self.deconv2 = nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(ngf)
        self.conv4 = nn.Conv2d(ngf, out_channels, kernel_size=7, padding=3, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.res_blocks(x)

        x = self.deconv1(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.tanh(x)
        return x

class GeneratorB2A(nn.Module):
    """G_B2A：RGB图 → 增强RGB图（图像增强生成器，阶段3启用）"""
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_res_blocks=9):
        super(GeneratorB2A, self).__init__()
        # 下采样
        self.conv1 = nn.Conv2d(in_channels, ngf, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*2)
        self.conv3 = nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*4)

        # 残差块
        self.res_blocks = nn.Sequential(*[ResidualBlock(ngf*4) for _ in range(n_res_blocks)])

        # 上采样
        self.deconv1 = nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf*2)
        self.deconv2 = nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(ngf)
        self.conv4 = nn.Conv2d(ngf, out_channels, kernel_size=7, padding=3, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.res_blocks(x)

        x = self.deconv1(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.tanh(x)
        return x

class DiscriminatorA(nn.Module):
    """DiscriminatorA：判别深度图（真实/补全）"""
    def __init__(self, in_channels=1, ndf=64, n_layers=3):
        super(DiscriminatorA, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        mult = 1
        for i in range(1, n_layers):
            prev_mult = mult
            mult = min(2**i, 8)
            layers.append(nn.Conv2d(ndf*prev_mult, ndf*mult, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(ndf*mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        prev_mult = mult
        mult = min(2**n_layers, 8)
        layers.append(nn.Conv2d(ndf*prev_mult, ndf*mult, kernel_size=4, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(ndf*mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(ndf*mult, 1, kernel_size=4, stride=1, padding=1, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DiscriminatorB(nn.Module):
    """DiscriminatorB：判别RGB图（真实/增强），阶段3启用"""
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super(DiscriminatorB, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        mult = 1
        for i in range(1, n_layers):
            prev_mult = mult
            mult = min(2**i, 8)
            layers.append(nn.Conv2d(ndf*prev_mult, ndf*mult, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(ndf*mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        prev_mult = mult
        mult = min(2**n_layers, 8)
        layers.append(nn.Conv2d(ndf*prev_mult, ndf*mult, kernel_size=4, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(ndf*mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(ndf*mult, 1, kernel_size=4, stride=1, padding=1, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)