import os
import re
import random
import gc
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from transformers import SamModel, SamProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading model from Hugging Face...")
model = SamModel.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model.to(device)
print("Model loaded successfully!")

# 冻结所有参数，只训练mask decoder
for name, param in model.named_parameters():
    param.requires_grad = False
for name, param in model.mask_decoder.named_parameters():
    param.requires_grad = True

# 打印可训练参数数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params}/{total_params} ({trainable_params/total_params*100:.2f}%)")

def extract_base_filename(filename):
    """从文件名中提取基础部分"""
    base = os.path.splitext(filename)[0]
    base = re.sub(r'_label_binary$', '', base)
    return base

class EnhancedSAMDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor, is_training=True, augment=True, max_samples=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.is_training = is_training
        self.augment = augment

        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png') and '_label_binary' in f]

        self.mask_map = {}
        for mask_file in self.mask_files:
            base = extract_base_filename(mask_file)
            self.mask_map[base] = mask_file

        self.valid_images = []
        for img_file in self.image_files:
            base = extract_base_filename(img_file)
            if base in self.mask_map:
                self.valid_images.append(img_file)

        # 限制图像数量
        if max_samples is not None and len(self.valid_images) > max_samples:
            self.valid_images = self.valid_images[:max_samples]
            print(f"Limited to first {max_samples} images")

        print(f"Found {len(self.valid_images)} valid image-mask pairs")
        if len(self.valid_images) == 0:
            print("Available image files:", self.image_files[:5])
            print("Available mask files:", self.mask_files[:5])
            raise ValueError("No valid image-mask pairs found!")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        try:
            img_file = self.valid_images[idx]
            img_path = os.path.join(self.image_dir, img_file)
            image = Image.open(img_path).convert('RGB')

            base = extract_base_filename(img_file)
            mask_file = self.mask_map[base]
            mask_path = os.path.join(self.mask_dir, mask_file)
            mask = Image.open(mask_path)
            mask_array = np.array(mask)

            # 数据增强
            if self.is_training and self.augment:
                if random.random() > 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    mask_array = np.fliplr(mask_array)
                if random.random() > 0.5:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    mask_array = np.flipud(mask_array)
                if random.random() > 0.5:
                    angle = random.uniform(-15, 15)
                    image = image.rotate(angle, resample=Image.BILINEAR)
                    mask_array = np.array(Image.fromarray(mask_array).rotate(angle, resample=Image.NEAREST))
                brightness_factor = random.uniform(0.8, 1.2)
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness_factor)
                contrast_factor = random.uniform(0.8, 1.2)
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast_factor)
                if random.random() > 0.8:
                    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

            if mask_array.max() > 1:
                mask_binary = (mask_array > 127).astype(np.uint8)
            else:
                mask_binary = (mask_array > 0.5).astype(np.uint8)

            y_idx, x_idx = np.where(mask_binary > 0)
            if len(x_idx) > 0 and len(y_idx) > 0:
                x_min, x_max = np.min(x_idx), np.max(x_idx)
                y_min, y_max = np.min(y_idx), np.max(y_idx)
                padding = 5
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(mask_array.shape[1] - 1, x_max + padding)
                y_max = min(mask_array.shape[0] - 1, y_max + padding)
                bbox = [[x_min, y_min, x_max, y_max]]
            else:
                h, w = mask_array.shape
                bbox = [[w//4, h//4, 3*w//4, 3*h//4]]

            inputs = self.processor(
                image,
                input_boxes=[bbox],
                return_tensors="pt"
            )
            for key in inputs:
                inputs[key] = inputs[key].squeeze(0)

            mask_pil = Image.fromarray(mask_binary * 255)
            mask_resized = mask_pil.resize((256, 256), Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_resized)).float() / 255.0

            return inputs, mask_tensor

        except Exception as e:
            print(f"Error processing {img_file if 'img_file' in locals() else 'unknown'}: {e}")
            return None

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None, None
        inputs_list, masks_list = zip(*batch)
        batch_inputs = {}
        for key in inputs_list[0].keys():
            batch_inputs[key] = torch.stack([item[key] for item in inputs_list])
        batch_masks = torch.stack(masks_list)
        return batch_inputs, batch_masks

# 设置数据路径
image_dir = "/root/train/01-Mine/Image"
mask_dir = "/root/Label_Binary"

# 创建数据集，限制为516张图片
full_dataset = EnhancedSAMDataset(image_dir, mask_dir, processor, is_training=True, max_samples=516)

# 训练/验证分割
train_indices, val_indices = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,
    random_state=42,
    shuffle=True
)
train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=full_dataset.collate_fn,
    num_workers=2,
    pin_memory=True
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=full_dataset.collate_fn,
    num_workers=1,
    pin_memory=True
)
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# 修复缺失的dropout_p属性
for name, module in model.named_modules():
    if 'Attention' in module.__class__.__name__:
        if not hasattr(module, 'dropout_p'):
            module.dropout_p = 0.0

# 组合损失函数
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.to(targets.device)
        bce_loss = self.bce(inputs, targets)
        inputs_sigmoid = torch.sigmoid(inputs).view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_sigmoid * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_sigmoid.sum() + targets_flat.sum() + smooth)
        return bce_loss + dice_loss

criterion = DiceBCELoss()

optimizer = optim.AdamW(
    model.mask_decoder.parameters(),
    lr=1e-5,
    weight_decay=0.01
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*10, eta_min=1e-7)

# 添加scaler用于混合精度训练
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

def calculate_sam_loss(outputs, batch_masks, criterion):
    pred_masks = outputs.pred_masks
    iou_scores = outputs.iou_scores
    pred_masks = pred_masks.to(batch_masks.device)
    iou_scores = iou_scores.to(batch_masks.device)
    batch_size = pred_masks.shape[0]
    pred_masks = pred_masks.squeeze(1)
    iou_scores = iou_scores.squeeze(1)
    total_loss = 0
    for i in range(batch_size):
        loss1 = criterion(pred_masks[i, 0].unsqueeze(0), batch_masks[i].unsqueeze(0))
        loss2 = criterion(pred_masks[i, 1].unsqueeze(0), batch_masks[i].unsqueeze(0))
        loss3 = criterion(pred_masks[i, 2].unsqueeze(0), batch_masks[i].unsqueeze(0))
        weights = torch.softmax(iou_scores[i], dim=0)
        weighted_loss = weights[0] * loss1 + weights[1] * loss2 + weights[2] * loss3
        total_loss += weighted_loss
    return total_loss / batch_size

def calculate_iou(pred, target):
    target_device = target.device
    pred = pred.to(target_device)
    pred = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0:
        return torch.tensor(1.0, device=target_device, dtype=torch.float32)
    return intersection / union

num_epochs = 50
best_iou = 0.0
patience = 8
counter = 0
accumulation_steps = 2

def check_data_samples(dataloader, num_samples=3):
    print("Checking data samples...")
    for i, (inputs, masks) in enumerate(dataloader):
        if i >= num_samples:
            break
        print(f"Sample {i+1}:")
        print(f" Input keys: {list(inputs.keys())}")
        print(f" Input shapes: {[v.shape for v in inputs.values()]}")
        print(f" Mask shape: {masks.shape}")
        print(f" Mask unique values: {torch.unique(masks)}")
        print(f" Mask non-zero pixels: {(masks > 0).sum().item()}/{masks.numel()}")
        print()

check_data_samples(train_dataloader)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    batch_count = 0
    train_iou = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    optimizer.zero_grad()
    
    for i, (batch_inputs, batch_masks) in enumerate(progress_bar):
        if batch_inputs is None:
            continue
            
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        batch_masks = batch_masks.to(device)
        
        # 修正：移除torch.no_grad()，使用正确的混合精度训练
        if device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(**batch_inputs)
                loss = calculate_sam_loss(outputs, batch_masks, criterion)
        else:
            outputs = model(**batch_inputs)
            loss = calculate_sam_loss(outputs, batch_masks, criterion)
            
        loss = loss / accumulation_steps
        
        if epoch == 0 and i == 0:
            print("First batch outputs:")
            print(f" pred_masks shape: {outputs.pred_masks.shape}")
            print(f" pred_masks device: {outputs.pred_masks.device}")
            print(f" batch_masks device: {batch_masks.device}")
            print(f" iou_scores shape: {outputs.iou_scores.shape}")
            print(f" iou_scores values: {outputs.iou_scores[0, 0]}")
        
        pred_masks = outputs.pred_masks.squeeze(1)
        iou_scores = outputs.iou_scores.squeeze(1)
        pred_masks = pred_masks.to(batch_masks.device)
        
        batch_iou = 0
        count = 0
        for j in range(pred_masks.size(0)):
            best_mask_idx = torch.argmax(iou_scores[j])
            best_pred_mask = pred_masks[j, best_mask_idx]
            iou_val = calculate_iou(best_pred_mask, batch_masks[j])
            batch_iou += iou_val.item()
            count += 1
        
        if count > 0:
            batch_iou = batch_iou / count
        else:
            batch_iou = 0
            
        train_iou += batch_iou
        
        # 修正：使用scaler进行反向传播
        if device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            if device.type == 'cuda':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.mask_decoder.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.mask_decoder.parameters(), max_norm=1.0)
                optimizer.step()
                
            optimizer.zero_grad()
            scheduler.step()
        
        epoch_loss += loss.item() * accumulation_steps
        batch_count += 1
        
        progress_bar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'iou': f'{batch_iou:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        if i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # 处理剩余的梯度
    if len(train_dataloader) % accumulation_steps != 0:
        if device.type == 'cuda':
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.mask_decoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.mask_decoder.parameters(), max_norm=1.0)
            optimizer.step()
            
        optimizer.zero_grad()
        scheduler.step()
    
    model.eval()
    val_loss = 0
    val_batch_count = 0
    val_iou = 0
    
    with torch.no_grad():
        val_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        for batch_inputs, batch_masks in val_bar:
            if batch_inputs is None:
                continue
                
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            batch_masks = batch_masks.to(device)
            
            outputs = model(**batch_inputs)
            loss = calculate_sam_loss(outputs, batch_masks, criterion)
            
            pred_masks = outputs.pred_masks.squeeze(1)
            iou_scores = outputs.iou_scores.squeeze(1)
            pred_masks = pred_masks.to(batch_masks.device)
            
            batch_iou = 0
            count = 0
            for j in range(pred_masks.size(0)):
                best_mask_idx = torch.argmax(iou_scores[j])
                best_pred_mask = pred_masks[j, best_mask_idx]
                iou_val = calculate_iou(best_pred_mask, batch_masks[j])
                batch_iou += iou_val.item()
                count += 1
                
            if count > 0:
                batch_iou = batch_iou / count
            else:
                batch_iou = 0
                
            val_iou += batch_iou
            val_loss += loss.item()
            val_batch_count += 1
            
            val_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{batch_iou:.4f}'
            })
    
    if batch_count > 0 and val_batch_count > 0:
        avg_train_loss = epoch_loss / batch_count
        avg_val_loss = val_loss / val_batch_count
        avg_train_iou = train_iou / batch_count
        avg_val_iou = val_iou / val_batch_count
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")
        
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.mask_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_val_loss,
                'iou': best_iou,
                'config': model.config
            }, "original_train.pth")
            print(f"Saved best model with validation IoU: {best_iou:.4f}")
        else:
            counter += 1
            print(f"Validation IoU didn't improve for {counter} epochs")
            
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    else:
        print(f"Epoch {epoch+1}: No valid batches")
    
    gc.collect()
    torch.cuda.empty_cache()

# 最终保存也使用original_train.pth
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.mask_decoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'config': model.config
}, "original_train_final.pth")

print("Training completed!")
print(f"Best validation IoU: {best_iou:.4f}")