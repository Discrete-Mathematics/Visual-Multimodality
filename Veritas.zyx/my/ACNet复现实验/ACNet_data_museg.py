import numpy as np
import os
from torch.utils.data import Dataset
import skimage.transform
import random
import torchvision
import torch
from PIL import Image
import glob
import matplotlib
import matplotlib.colors

# MUSeg数据集参数
MUSEG_IMAGE_H = 1536  # MUSeg原始高度
MUSEG_IMAGE_W = 2048  # MUSeg原始宽度
TARGET_H = 512  # 调整为适合训练的尺寸
TARGET_W = 512



class RandomRotation(object):
    """自定义随机旋转变换，处理字典数据"""

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        from torchvision import transforms as T

        # 创建随机旋转变换
        rotation = T.RandomRotation(self.degrees)

        # 对RGB图像应用旋转
        sample['image'] = rotation(sample['image'])

        # 注意：深度图和标签通常不进行旋转，因为旋转会改变几何关系
        # 如果你确实需要旋转深度图和标签，可以使用以下代码：
        # sample['depth'] = rotation(sample['depth'])
        # sample['label'] = rotation(sample['label'])

        return sample


class RandomResizedCrop(object):
    """自定义随机缩放裁剪变换，处理字典数据"""

    def __init__(self, size, scale):
        self.size = size
        self.scale = scale

    def __call__(self, sample):
        from torchvision import transforms as T

        # 创建随机缩放裁剪变换
        resized_crop = T.RandomResizedCrop(self.size, scale=self.scale)

        # 对RGB图像应用变换
        sample['image'] = resized_crop(sample['image'])

        # 深度图和标签通常不进行这种变换
        # 如果需要，可以添加类似的代码处理深度图和标签

        return sample
class MUSegDataset(Dataset):
    def __init__(self, transform=None, phase_train=True, data_root='/path/to/MUSeg'):
        self.phase_train = phase_train
        self.transform = transform
        self.data_root = data_root

        # MUSeg数据集结构
        if phase_train:
            self.image_dir = os.path.join(data_root, 'RGB')
            self.depth_dir = os.path.join(data_root, 'Depth')
            self.label_dir = os.path.join(data_root, 'Label')
        else:
            self.image_dir = os.path.join(data_root, 'RGB')
            self.depth_dir = os.path.join(data_root, 'Depth')
            self.label_dir = os.path.join(data_root, 'Label')

        # 读取train.txt或val.txt获取文件列表
        if phase_train:
            list_file = os.path.join(data_root, 'train.txt')
        else:
            list_file = os.path.join(data_root, 'val.txt')

        # 从txt文件中读取文件名
        with open(list_file, 'r') as f:
            file_names = [line.strip() for line in f.readlines()]

        # RGB文件使用.jpg扩展名，Depth使用.png扩展名，Label使用_label.png扩展名
        self.image_files = [os.path.join(self.image_dir, fname + '.jpg') for fname in file_names]
        self.depth_files = [os.path.join(self.depth_dir, fname + '.png') for fname in file_names]
        self.label_files = [os.path.join(self.label_dir, fname + '_label.png') for fname in file_names]

        # 确保文件存在
        self.image_files = [f for f in self.image_files if os.path.exists(f)]
        self.depth_files = [f for f in self.depth_files if os.path.exists(f)]
        self.label_files = [f for f in self.label_files if os.path.exists(f)]

        print(
            f"有效文件数量 - RGB: {len(self.image_files)}, Depth: {len(self.depth_files)}, Label: {len(self.label_files)}")

    def __len__(self):
        return min(len(self.image_files), len(self.depth_files), len(self.label_files))

    def __getitem__(self, idx):
        # 确保索引在有效范围内
        idx = idx % len(self)

        # 读取图像
        image = np.array(Image.open(self.image_files[idx]))  # (H, W, 3)
        depth = np.array(Image.open(self.depth_files[idx]))  # (H, W)
        label = np.array(Image.open(self.label_files[idx]))  # (H, W)

        # 确保深度图是单通道
        if len(depth.shape) > 2:
            depth = depth[:, :, 0]

        sample = {'image': image, 'depth': depth, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomHSV(object):
    """
    随机调整HSV颜色空间 - 修复版
    """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']

        # 将图像从 [0, 255] 缩放到 [0, 1]
        img_normalized = img / 255.0

        img_hsv = matplotlib.colors.rgb_to_hsv(img_normalized)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]

        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))

        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random / 255.0, 0, 1)  # v_random 也需要调整

        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        # 将图像从 [0, 1] 缩放回 [0, 255]
        img_new = (img_new * 255.0).astype(np.uint8)

        return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}


class RandomScale(object):
    """
    随机缩放
    """

    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = skimage.transform.resize(image, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomFlip(object):
    def __call__(self, sample):
        # 确保处理字典
        if random.random() < 0.5:
            image = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
            depth = sample['depth'].transpose(Image.FLIP_LEFT_RIGHT)
            label = sample['label'].transpose(Image.FLIP_LEFT_RIGHT)
            return {'image': image, 'depth': depth, 'label': label}
        return sample


class ScaleNorm(object):
    """
    缩放和标准化图像尺寸到固定大小
    """

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Bi-linear 插值用于图像
        image = skimage.transform.resize(image, (TARGET_H, TARGET_W), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor 插值用于深度图和标签（保持边缘清晰）
        depth = skimage.transform.resize(depth, (TARGET_H, TARGET_W), order=0,
                                         mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (TARGET_H, TARGET_W), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomCrop(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # 获取图像尺寸
        w, h = image.size

        # 如果图像比目标尺寸小，先调整大小
        if w < self.width or h < self.height:
            # 调整到最小能包含目标尺寸的大小
            scale = max(self.width / w, self.height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)
            depth = depth.resize((new_w, new_h), Image.NEAREST)
            label = label.resize((new_w, new_h), Image.NEAREST)
            w, h = new_w, new_h

        # 随机生成裁剪的左上角坐标
        x1 = random.randint(0, w - self.width)
        y1 = random.randint(0, h - self.height)

        # 对RGB图像、深度图和标签进行同样的裁剪
        cropped_image = image.crop((x1, y1, x1 + self.width, y1 + self.height))
        cropped_depth = depth.crop((x1, y1, x1 + self.width, y1 + self.height))
        cropped_label = label.crop((x1, y1, x1 + self.width, y1 + self.height))

        return {'image': cropped_image, 'depth': cropped_depth, 'label': cropped_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Generate different label scales for multi-scale supervision
        label2 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
                                          order=0, mode='reflect', preserve_range=True)
        label3 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
                                          order=0, mode='reflect', preserve_range=True)
        label4 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
                                          order=0, mode='reflect', preserve_range=True)
        label5 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
                                          order=0, mode='reflect', preserve_range=True)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(float)

        # 转换为Tensor
        image_tensor = torch.from_numpy(image).float()
        depth_tensor = torch.from_numpy(depth).float()
        label_tensor = torch.from_numpy(label).float()
        label2_tensor = torch.from_numpy(label2).float()
        label3_tensor = torch.from_numpy(label3).float()
        label4_tensor = torch.from_numpy(label4).float()
        label5_tensor = torch.from_numpy(label5).float()

        return {
            'image': image_tensor,
            'depth': depth_tensor,
            'label': label_tensor,
            'label2': label2_tensor,
            'label3': label3_tensor,
            'label4': label4_tensor,
            'label5': label5_tensor
        }



class Normalize(object):
    """
    归一化处理 - 修复版，处理Tensor输入
    """

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        # 图像归一化 (已经是Tensor)
        image = image / 255.0  # 转换为 [0, 1]
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)

        # 深度图归一化 - 需要根据你的数据调整均值和标准差
        # 假设深度图已经是合适的范围，如果需要可以调整
        depth = torchvision.transforms.Normalize(mean=[0.5], std=[0.5])(depth)

        sample['image'] = image
        sample['depth'] = depth

        return sample