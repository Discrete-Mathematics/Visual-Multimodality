import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class SegmentationDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(SegmentationDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   从文件中读取RGB图像和深度图像
        #-------------------------------#
        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".jpg"))
        depth       = Image.open(os.path.join(os.path.join(self.dataset_path, "DepthImages"), name + ".png"))  # 假设深度图像是PNG格式
        png         = Image.open(os.path.join(os.path.join(self.dataset_path, "SegmentationClass"), name + "_label.png"))

        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg, depth, png = self.get_random_data(jpg, depth, png, self.input_shape, random=self.train)

        #-------------------------------#
        #   预处理RGB图像
        #-------------------------------#
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])

        #-------------------------------#
        #   预处理深度图像
        #-------------------------------#
        depth = np.array(depth, np.float32)  # 将深度图像转换为 NumPy 数组
        # 判断通道数
        if len(depth.shape) == 2:  # 单通道图像，形状为 [H, W]
            depth = np.stack([depth] * 3, axis=0)  # 复制为 3 通道，形状变为 [3, H, W]
        elif len(depth.shape) == 3 and depth.shape[2] == 3:  # 3 通道图像，形状为 [H, W, 3]
            depth = np.transpose(depth, [2, 0, 1])  # 调整为 [3, H, W]
        else:
            raise ValueError(f"Unsupported depth image shape: {depth.shape}")
        #-------------------------------#
        #   处理标签
        #-------------------------------#
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, depth, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, depth, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        depth = Image.fromarray(np.array(depth))  # 深度图像也需要转换为PIL格式
        label = Image.fromarray(np.array(label))

        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            depth = depth.resize((nw, nh), Image.NEAREST)  # 深度图像使用最近邻插值
            label = label.resize((nw, nh), Image.NEAREST)

            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_depth = Image.new('L', [w, h], (0))  # 深度图像是单通道的
            new_label = Image.new('L', [w, h], (0))

            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
            new_depth.paste(depth, ((w - nw) // 2, (h - nh) // 2))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))

            return new_image, new_depth, new_label

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        depth = depth.resize((nw, nh), Image.NEAREST)
        label = label.resize((nw, nh), Image.NEAREST)

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_depth = Image.new('L', (w, h), (0))
        new_label = Image.new('L', (w, h), (0))

        new_image.paste(image, (dx, dy))
        new_depth.paste(depth, (dx, dy))
        new_label.paste(label, (dx, dy))

        image = new_image
        depth = new_depth
        label = new_label

        image_data = np.array(image, np.uint8)
        depth_data = np.array(depth, np.uint8)

        #------------------------------------------#
        #   高斯模糊
        #------------------------------------------#
        blur = self.rand() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)
            depth_data = cv2.GaussianBlur(depth_data, (5, 5), 0)

        #------------------------------------------#
        #   旋转
        #------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            depth_data = cv2.warpAffine(depth_data, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, depth_data, label


# DataLoader中collate_fn使用
def seg_dataset_collate(batch):
    images = []
    depths = []
    pngs = []
    seg_labels = []
    for img, depth, png, labels in batch:
        images.append(img)
        depths.append(depth)
        pngs.append(png)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    depths = torch.from_numpy(np.array(depths)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, depths, pngs, seg_labels