# complete_fixed_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import time
import sys
import torch.nn.functional as F
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ACNet_data_museg import MUSegDataset, ScaleNorm, RandomCrop, RandomFlip, ToTensor, Normalize
from ACNet_models_V1 import ACNet
from utils import AverageMeter, print_log


class SimpleTransform(object):
    """处理 numpy 数组数据的简单变换"""

    def __init__(self, phase_train=True):
        self.phase_train = phase_train

    def __call__(self, sample):
        from torchvision import transforms as T
        import torchvision.transforms.functional as F
        from PIL import Image

        # 确保样本是字典格式
        if not isinstance(sample, dict):
            raise ValueError(f"样本应该是字典格式，但得到的是 {type(sample)}")

        image, depth, label = sample['image'], sample['depth'], sample['label']

        # 🎯 处理 numpy 数组 - 转换为 PIL 图像
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        if isinstance(depth, np.ndarray):
            depth = Image.fromarray(depth.astype('uint8'))
        if isinstance(label, np.ndarray):
            label = Image.fromarray(label.astype('uint8'))

        # 基础变换 - 调整大小到固定尺寸
        target_size = (512, 512)

        # 调整图像大小
        image = F.resize(image, target_size, interpolation=T.InterpolationMode.BILINEAR)
        depth = F.resize(depth, target_size, interpolation=T.InterpolationMode.NEAREST)
        label = F.resize(label, target_size, interpolation=T.InterpolationMode.NEAREST)

        # 转换为张量
        image = F.to_tensor(image)
        depth = F.to_tensor(depth)
        label = torch.from_numpy(np.array(label)).long()

        # 归一化图像（使用ImageNet统计量）
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return {'image': image, 'depth': depth, 'label': label}
# 🎯 在导入部分之后，类定义之前添加 FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
class CompleteFixedACNetExperiment:
    def __init__(self):
        # 训练参数 - 基于实际测试结果优化
        self.BATCH_SIZE = 2  # 测试显示可以支持2
        self.NUM_EPOCHS = 100
        self.LEARNING_RATE = 1e-4
        self.NUM_CLASSES = 16
        self.DATA_ROOT = 'F:/大创-视觉多模态/MUSeg'
        self.CHECKPOINT_DIR = './complete_fixed_checkpoints'
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {self.device}')
        print(f'Batch Size: {self.BATCH_SIZE}')

        # 实验记录
        self.experiment_log = {
            'config': {
                'batch_size': self.BATCH_SIZE,
                'epochs': self.NUM_EPOCHS,
                'learning_rate': self.LEARNING_RATE,
                'num_classes': self.NUM_CLASSES
            }
        }

        # 初始化
        self.setup_data()
        self.setup_model()

    def setup_data(self):
        """设置数据加载器 - 使用全新的简单变换"""
        print("设置数据加载器...")

        # 使用全新的简单变换
        train_transform = SimpleTransform(phase_train=True)
        val_transform = SimpleTransform(phase_train=False)

        try:
            self.train_dataset = MUSegDataset(transform=train_transform, phase_train=True, data_root=self.DATA_ROOT)
            self.val_dataset = MUSegDataset(transform=val_transform, phase_train=False, data_root=self.DATA_ROOT)

            print(f"训练集大小: {len(self.train_dataset)}")
            print(f"验证集大小: {len(self.val_dataset)}")

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=False
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=False
            )

            print(f"实际batch数: {len(self.train_loader)}")

        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            raise

    def setup_model(self):
        """设置模型 - 修复版"""
        print("初始化ACNet模型...")
        try:
            self.model = ACNet(num_class=self.NUM_CLASSES, pretrained=True)
            self.model = self.model.to(self.device)

            self.criterion = nn.CrossEntropyLoss(ignore_index=255)

            # 🎯 使用 AdamW 优化器（去掉 momentum 参数）
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.LEARNING_RATE,  # 使用原来的学习率
                weight_decay=1e-4
            )

            # 🎯 使用 OneCycleLR 学习率调度器
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=0.001,
                epochs=self.NUM_EPOCHS,
                steps_per_epoch=len(self.train_loader)
            )

            print("✅ 模型初始化完成")

        except Exception as e:
            print(f"模型初始化失败: {str(e)}")
            raise

    def get_main_output(self, outputs):
        """获取主输出 - 处理多尺度输出"""
        if isinstance(outputs, tuple):
            return outputs[0]  # 返回第一个输出（主输出）
        else:
            return outputs

    def train_epoch(self, epoch, global_step):
        """训练一个epoch - 完整修复版"""
        if not hasattr(self, 'criterion'):
            raise AttributeError("criterion 未初始化! 请检查 setup_model 方法")

        if not hasattr(self, 'model'):
            raise AttributeError("model 未初始化! 请检查 setup_model 方法")

        if not hasattr(self, 'optimizer'):
            raise AttributeError("optimizer 未初始化! 请检查 setup_model 方法")
        self.model.train()
        losses = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        end = time.time()
        print(f"开始训练epoch {epoch}, 总batch数: {len(self.train_loader)}")

        for i, sample in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            # 移动数据到设备
            rgb = sample['image'].to(self.device)
            depth = sample['depth'].to(self.device)
            label = sample['label'].to(self.device)

            # 前向传播
            outputs = self.model(rgb, depth)

            # 获取主输出
            main_output = self.get_main_output(outputs)

            # 计算损失 - 使用标准交叉熵损失
            loss = self.criterion(main_output, label.long())

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), rgb.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            global_step += 1

            # 每20个batch打印一次
            if i % 20 == 0 or i == len(self.train_loader) - 1:
                print_log(global_step, epoch, i + 1, len(self.train_loader), len(self.train_loader.dataset),
                          losses.avg, batch_time.avg)

        print(f"完成训练epoch {epoch}")
        return global_step, losses.avg

    def validate_epoch(self, epoch):
        """验证一个epoch - 修复版"""
        self.model.eval()
        losses = AverageMeter()

        # 用于计算指标
        total_correct = 0
        total_pixels = 0

        with torch.no_grad():
            for i, sample in enumerate(self.val_loader):
                rgb = sample['image'].to(self.device)
                depth = sample['depth'].to(self.device)
                label = sample['label'].to(self.device)

                output = self.model(rgb, depth)
                main_output = self.get_main_output(output)
                loss = self.criterion(main_output, label.long())
                losses.update(loss.item(), rgb.size(0))

                # 计算准确率
                pred = torch.argmax(main_output, dim=1)
                correct = (pred == label).sum().item()
                total_correct += correct
                total_pixels += label.numel()

        accuracy = total_correct / total_pixels if total_pixels > 0 else 0

        print(f'验证结果 - Epoch {epoch}:')
        print(f'  损失: {losses.avg:.4f}')
        print(f'  准确率: {accuracy:.4f}')

        return {
            'loss': losses.avg,
            'accuracy': accuracy
        }

    def run_experiment(self):
        """运行完整修复实验"""
        print("=== 完整修复ACNet实验开始 ===")
        print(f"实验配置: {self.experiment_log['config']}")

        best_accuracy = 0.0
        global_step = 0
        patience = 5  # 连续5个epoch没有改进就停止
        patience_counter = 0

        try:
            for epoch in range(self.NUM_EPOCHS):
                print(f'\nEpoch {epoch + 1}/{self.NUM_EPOCHS}')
                print('-' * 50)

                # 训练一个epoch
                global_step, train_loss = self.train_epoch(epoch, global_step)

                # 验证
                val_results = None  # 初始化变量
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    val_results = self.validate_epoch(epoch)

                    # 记录最佳模型
                    if val_results['accuracy'] > best_accuracy:
                        best_accuracy = val_results['accuracy']
                        patience_counter = 0  # 重置计数器
                        torch.save(self.model.state_dict(),
                                   os.path.join(self.CHECKPOINT_DIR, 'best_model.pth'))
                        print(f'  🎯 新的最佳模型已保存! 准确率: {best_accuracy:.4f}')
                    else:
                        patience_counter += 1  # 没有改进，计数器+1

                    self.experiment_log.setdefault('training_history', []).append({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        **val_results
                    })

                    # 早停检查（只在验证时检查）
                    if patience_counter >= patience:
                        print(f"🎯 早停触发! 连续 {patience} 个epoch准确率没有提升")
                        break

                # 更新学习率
                self.scheduler.step()

                # 每10个epoch保存一次检查点
                if (epoch + 1) % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                    }, os.path.join(self.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch + 1}.pth'))

        except Exception as e:
            print(f"训练过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()

        # 生成最终报告
        self.generate_report(best_accuracy)

    def generate_report(self, best_accuracy):
        """生成实验报告"""
        print("\n" + "=" * 60)
        print("完整修复ACNet实验报告")
        print("=" * 60)

        print(f"\n📊 最终结果:")
        print(f"  最佳验证准确率: {best_accuracy:.4f}")

        print(f"\n⚙️ 实验配置:")
        for key, value in self.experiment_log['config'].items():
            print(f"  {key}: {value}")

        print(f"\n💾 模型保存位置: {self.CHECKPOINT_DIR}")



if __name__ == '__main__':
    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    experiment = CompleteFixedACNetExperiment()
    experiment.run_experiment()