import numpy as np
from torch import nn
import torch
import os
import torch.nn.functional as F
from torch.autograd import Variable

# MUSeg的类别权重（暂时使用均匀权重）
museg_frq = [1.0] * 16

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# MUSeg的16类别颜色映射
label_colours = [
    (0, 0, 0),           # 0=background
    (148, 65, 137),      # 1=rock
    (255, 116, 69),      # 2=support
    (86, 156, 137),      # 3=ventilation
    (202, 179, 158),     # 4=pipeline
    (155, 99, 235),      # 5=conveyor
    (161, 107, 108),     # 6=equipment
    (133, 160, 103),     # 7=mineral
    (76, 152, 126),      # 8=vehicle
    (84, 62, 35),        # 9=person
    (44, 80, 130),       # 10=cable
    (31, 184, 157),      # 11=lighting
    (101, 144, 77),      # 12=sign
    (23, 197, 62),       # 13=water
    (141, 168, 145),     # 14=debris
    (142, 151, 136)      # 15=other
]


class CrossEntropyLoss2d_eval(nn.Module):
    def __init__(self, weight=museg_frq, device=None):
        super(CrossEntropyLoss2d_eval, self).__init__()
        self.device = device
        # 将权重转换为tensor，但不立即指定设备
        self.weight_tensor = torch.from_numpy(np.array(weight)).float()
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight_tensor,
                                           reduction='none')

    def forward(self, inputs_scales, targets_scales):
        losses = []
        inputs = inputs_scales
        targets = targets_scales
        # for inputs, targets in zip(inputs_scales, targets_scales):
        mask = targets > 0
        targets_m = targets.clone()
        targets_m[mask] -= 1

        # 确保权重在与输入相同的设备上
        if self.weight_tensor.device != inputs.device:
            self.weight_tensor = self.weight_tensor.to(inputs.device)
            self.ce_loss = nn.CrossEntropyLoss(weight=self.weight_tensor,
                                               reduction='none')

        loss_all = self.ce_loss(inputs, targets_m.long())
        losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss


# 在 utils.py 中修复 CrossEntropyLoss2d
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=museg_frq, device=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.device = device
        # 将权重转换为tensor
        self.weight_tensor = torch.from_numpy(np.array(weight)).float()
        if device:
            self.weight_tensor = self.weight_tensor.to(device)
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.weight_tensor,
            reduction='mean',
            ignore_index=255  # 忽略无效标签
        )

    def forward(self, inputs_scales, targets_scales):
        losses = []

        # 确保 inputs_scales 是列表
        if not isinstance(inputs_scales, list):
            inputs_scales = [inputs_scales]

        # 确保 targets_scales 是列表
        if not isinstance(targets_scales, list):
            targets_scales = [targets_scales]

        for inputs, targets in zip(inputs_scales, targets_scales):
            # 如果 inputs 是元组，取第一个元素（主输出）
            if isinstance(inputs, tuple):
                inputs = inputs[0]

            # 确保目标标签在正确范围内
            targets = targets.long()

            # 计算损失
            loss = self.ce_loss(inputs, targets)
            losses.append(loss)

        return sum(losses) / len(losses)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.loss = nn.NLLLoss(weight=torch.from_numpy(np.array(weight)).float(),
                                 reduction='none')

    def forward(self, input, target):
        return self.loss((1 - F.softmax(input, 1))**2 * F.log_softmax(input, 1), target)

class FocalLoss2d(nn.Module):
    def __init__(self, weight=museg_frq, gamma=0):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.fl_loss = FocalLoss(gamma=self.gamma, weight=self.weight, size_average=False)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            mask = targets > 0
            targets_m = targets.clone()
            targets_m[mask] -= 1
            loss_all = self.fl_loss(inputs, targets_m.long())
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss

def color_label_eval(label):
    colored_label = np.vectorize(lambda x: label_colours[int(x)])
    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()
    return colored.transpose([0, 2, 1])

def color_label(label):
    label = label.clone().cpu().data.numpy()
    colored_label = np.vectorize(lambda x: label_colours[int(x)])
    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()
    try:
        return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])


def print_log(step, epoch, batch_idx, total_batches, total_samples, loss, batch_time):
    """打印训练日志"""
    # 修改这里：使用batch数而不是样本数来计算百分比
    current_batch = batch_idx
    total_batch_count = total_batches  # 这应该是798

    # 计算基于batch的百分比
    percent = 100. * current_batch / total_batch_count

    print(f'Step: {step:5} Train Epoch: {epoch:3} [{current_batch:3}/{total_batch_count:3} ({percent:.1f}%)]\t'
          f'Loss: {loss:.6f} [{batch_time:.2f}s every {total_batch_count:3} data]')

def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch, local_count, num_train):
    epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'epoch': epoch_float,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}.pth".format(epoch_float)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))

def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        return step, epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred = imPred * (imLab > 0)

    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

def accuracy(preds, label):
    valid = (label > 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def macc(preds, label, num_class):
    a = np.zeros(num_class)
    b = np.zeros(num_class)
    for i in range(num_class):
        mask = (label == i+1)
        a_sum = (mask * preds == i+1).sum()
        b_sum = mask.sum()
        a[i] = a_sum
        b[i] = b_sum
    return a,b

class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

# 导出所有必要的函数和类
__all__ = [
    'CrossEntropyLoss2d',
    'CrossEntropyLoss2d_eval',
    'FocalLoss',
    'FocalLoss2d',
    'color_label_eval',
    'color_label',
    'print_log',
    'save_ckpt',
    'load_ckpt',
    'intersectionAndUnion',
    'accuracy',
    'macc',
    'AverageMeter',
    'museg_frq',
    'label_colours'
]