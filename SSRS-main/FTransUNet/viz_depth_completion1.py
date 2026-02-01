#!/usr/bin/env python
# viz_depth_completion_16bit.py
import os
import argparse
import json
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# ========== 项目 import ==========
from GAN_FTransUNet_train_val import EndToEndDepthCompletionSegmentation
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from gan import build_model
import lib.utils.configurator as cfg_file
load_config = cfg_file.load_config
cfg = cfg_file.cfg

# ---------- 参数 ----------
class TestConfig:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 1
        self.image_size = 256
        self.data_root  = "/root/autodl-tmp/dataset"
        self.gan_config = "rdf_cycle_patchgan_config.yaml"
        self.save_dir   = "./viz_test_depth1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',  default='./best_model.pth')
    parser.add_argument('--list',  default='./test.txt')
    parser.add_argument('--out',   default='./viz_test_depth')
    parser.add_argument('--n_show',type=int, default=50)
    parser.add_argument('--depth_min',type=float, default=0.0,  help='原始深度最小值（米或毫米）')
    parser.add_argument('--depth_max',type=float, default=65535.0,help='原始深度最大值（米或毫米）')
    return parser.parse_args()

# ---------- 深度→伪彩色（0-1 输入） ----------
def depth_to_jet(depth_tensor, vmin=0.0, vmax=1.0):
    depth_np = depth_tensor.squeeze().cpu().numpy()
    depth_norm = np.clip((depth_np - vmin) / (vmax - vmin + 1e-8), 0, 1)
    cmap = plt.get_cmap('jet')
    color = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    return color

# ---------- 数据集（16-bit 深度） ----------
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, ids, root_dir, transform=None, depth_min=0.0, depth_max=65535.0):
        self.ids = ids
        self.root_dir = root_dir
        self.transform = transform
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.img_fmt   = os.path.join(root_dir, "Image", "{}.jpg")
        self.depth_fmt = os.path.join(root_dir, "Depth", "{}.png")

    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        id_ = self.ids[idx]
    
        # ---- 1. 拼路径 ----
        img_path   = self.img_fmt.format(id_)
        depth_path = self.depth_fmt.format(id_)
    
        # ---- 2. 统一打印 ----
        print(f"[{id_}]  depth_path = {depth_path}")
    
        # ---- 3. RGB 用 PIL ----
        img = Image.open(img_path).convert('RGB')
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        img = img.resize((256, 256), Image.BILINEAR)
    
        # ---- 4. 深度用 cv2 读 16-bit ----
        depth16 = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth16 is None:
            raise FileNotFoundError(depth_path)
        depth16 = cv2.resize(depth16, (256, 256), interpolation=cv2.INTER_NEAREST)
    
        # ---- 5. 用同一路径再拿 PIL 对比 ----
        dsm_pil = Image.open(depth_path).resize((256, 256), Image.BILINEAR)
        dsm_arr = np.array(dsm_pil, dtype=np.float32)
    
        # ---- 6. 打印数值 ----
        print("opencv min/max:", depth16.min(), depth16.max())
        print("PIL    min/max:", dsm_arr.min(), dsm_arr.max())
    
        # ---- 7. 归一化到 0-1 ----
        depth_f = np.clip((depth16.astype(np.float32) - self.depth_min) /
                          (self.depth_max - self.depth_min + 1e-8), 0.0, 1.0)
        depth_tensor = torch.from_numpy(depth_f).unsqueeze(0)  # [1,H,W]
    
        # ---- 8. RGB tensor ----
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
    
        return id_, img_tensor, depth_tensor

# ---------- 主函数 ----------
def main():
    args = parse_args()
    cfg_test = TestConfig()
    device = cfg_test.device
    os.makedirs(args.out, exist_ok=True)

    # 1. 构造模型
    load_config(cfg, cfg_test.gan_config)
    depth_model = build_model(cfg, device)
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 16
    config_vit.n_skip = 3
    config_vit.patches.grid = (16, 16)
    seg_model = ViT_seg(config_vit, img_size=256, num_classes=16).to(device)
    model = EndToEndDepthCompletionSegmentation(
        depth_completion_model=depth_model,
        segmentation_model=seg_model
    ).to(device)

    # 2. 加载权重 + eval
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval(); model.depth_completion_model.eval(); seg_model.eval()

    # 3. 数据
    test_ids = [l.strip() for l in open(args.list).readlines() if l.strip()]
    test_dataset = TestDataset(test_ids, cfg_test.data_root,
                               transform=transforms.Compose([
                                   transforms.Resize((256, 256)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])
                               ]), depth_min=args.depth_min, depth_max=args.depth_max)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # 4. 推理 + 可视化
    count = 0
    with torch.no_grad():
        for file_id, rgb_tensor, depth_tensor in tqdm(test_loader, desc="Visualizing"):
            if count >= args.n_show: break
            rgb_tensor   = rgb_tensor.to(device)
            depth_tensor = depth_tensor.to(device)

            completed_depth, _ = model(rgb_tensor, depth_tensor)  # [B,1,H,W]

            # RGB 反归一化
            mean = rgb_tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = rgb_tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb_chw = rgb_tensor[0]
            rgb_np  = (rgb_chw * std + mean).clamp(0, 1).cpu().numpy()
            rgb_np  = np.transpose(rgb_np, (1, 2, 0))
            rgb_np  = (rgb_np * 255).astype(np.uint8)
            # 全局范围
            dmin = min(depth_tensor.min().item(), completed_depth.min().item())
            dmax = max(depth_tensor.max().item(), completed_depth.max().item())
            # 伪彩色（都用 0-1 区间）
            raw_color  = depth_to_jet(depth_tensor,  vmin=dmin, vmax=dmax)
            comp_color = depth_to_jet(completed_depth, vmin=dmin, vmax=dmax)

            canvas = np.hstack([rgb_np, raw_color, comp_color])
            Image.fromarray(canvas).save(os.path.join(args.out, f'{file_id[0]}_depth.png'))
            count += 1

    print(f'✅ 可视化完成，共 {count} 张 → {args.out}')

if __name__ == '__main__':
    main()