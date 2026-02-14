import random
import torch

def filter_and_sample_objects(gt_instances, max_num_objects=6,
                              min_score=0.3, min_volume=10):
    valid = []
    for obj in gt_instances:
        if hasattr(obj, 'score') and obj.score <= min_score:
            continue
        # 使用第一帧的mask检查体积
        if hasattr(obj, 'mask_sequence') and len(obj.mask_sequence) > 0:
            first_mask = obj.mask_sequence[0]
            if isinstance(first_mask, dict):
                num_pts = first_mask['pts'].sum().item()
            else:
                num_pts = first_mask.sum().item()
            if num_pts <= min_volume:
                continue
        valid.append(obj)

    if len(valid) > max_num_objects:
        valid = random.sample(valid, max_num_objects)
    return valid
 
def generate_occlusion_labels(gt_masks_per_frame, visibility_threshold=0.05):
    """生成遮挡标签
    如果某目标在某帧中可见点数占其总点数比例低于阈值，标记为遮挡
    
    Args:
        gt_masks_per_frame: list of (N_t,) bool tensors, 每帧的GT mask
        visibility_threshold: 可见性阈值
    Returns:
        occ_labels: (T,) bool tensor, True=被遮挡
    """
    total_points = max(m.sum().item() for m in gt_masks_per_frame)
    if total_points == 0:
        return torch.ones(len(gt_masks_per_frame), dtype=torch.bool)

    occ_labels = []
    for mask in gt_masks_per_frame:
        visible_ratio = mask.sum().item() / max(total_points, 1)
        occ_labels.append(visible_ratio < visibility_threshold)

    return torch.tensor(occ_labels, dtype=torch.bool)


def add_noise_to_mask(gt_mask, noise_ratio=0.05):
    """给GT mask添加噪声，模拟不完美的mask提示
    
    Args:
        gt_mask: dict of {modal_key: (1, 1, H, W) or (1, 1, N, 1) tensor}
                 与sam4d_base中mask_inputs格式一致（见_use_mask_as_output）
        noise_ratio: 翻转比例
    Returns:
        noisy_mask: 同格式的噪声mask
    """
    noisy_mask = {}
    for key, mask in gt_mask.items():
        m = mask.clone().float()
        flat = m.view(-1)
        num_flip = int(flat.numel() * noise_ratio)
        if num_flip > 0:
            flip_idx = torch.randperm(flat.numel(), device=mask.device)[:num_flip]
            flat[flip_idx] = 1.0 - flat[flip_idx]
        noisy_mask[key] = m
    return noisy_mask


def sample_points_from_mask(gt_mask, points_xyz=None, num_pos=1, num_neg=1, modal_info=None):
    """从GT mask中采样正/负提示点
    
    输出格式与 sam4d_infer.py add_new_points_or_box 中 point_inputs 一致:
    {
        'img': {'point_coords': (1, N, 2), 'point_labels': (1, N)},
        'pts': {'point_coords': (1, N, 3), 'point_labels': (1, N)},
    }
    
    Args:
        gt_mask: dict of {
            'img': (H, W) or (1, 1, H, W) bool/float tensor,
            'pts': (N_pts,) or (1, 1, N_pts, 1) bool/float tensor,
        }
        points_xyz: (N_pts, 3) tensor, 点云坐标（pts模态需要）
        num_pos: 正点数量
        num_neg: 负点数量
        modal_info: dict, 包含 image_size 等信息（用于img模态坐标缩放）
    Returns:
        point_inputs: dict，格式同推理时的 point_inputs
    """
    point_inputs = {}

    for key, mask in gt_mask.items():
        # 展平mask
        if mask.dim() == 4:  # (1, 1, H, W) or (1, 1, N, 1)
            flat_mask = mask.squeeze().view(-1).bool()
        elif mask.dim() == 2:
            flat_mask = mask.view(-1).bool()
        else:
            flat_mask = mask.view(-1).bool()

        pos_indices = torch.where(flat_mask)[0]
        neg_indices = torch.where(~flat_mask)[0]
        device = mask.device

        if key == 'img':
            H = mask.shape[-2] if mask.dim() >= 2 else int(mask.shape[0] ** 0.5)
            W = mask.shape[-1] if mask.dim() >= 2 else H

            # 采样正点像素坐标
            if len(pos_indices) > 0 and num_pos > 0:
                sel_pos = pos_indices[torch.randperm(len(pos_indices), device=device)[:num_pos]]
                pos_y = (sel_pos // W).float()
                pos_x = (sel_pos % W).float()
                pos_coords = torch.stack([pos_x, pos_y], dim=-1)  # (num_pos, 2) xy格式
            else:
                pos_coords = torch.zeros(0, 2, device=device)

            # 采样负点
            if len(neg_indices) > 0 and num_neg > 0:
                sel_neg = neg_indices[torch.randperm(len(neg_indices), device=device)[:num_neg]]
                neg_y = (sel_neg // W).float()
                neg_x = (sel_neg % W).float()
                neg_coords = torch.stack([neg_x, neg_y], dim=-1)
            else:
                neg_coords = torch.zeros(0, 2, device=device)

            coords = torch.cat([pos_coords, neg_coords], dim=0)  # (N, 2)
            labels = torch.cat([
                torch.ones(pos_coords.shape[0], dtype=torch.int32, device=device),
                torch.zeros(neg_coords.shape[0], dtype=torch.int32, device=device),
            ])

            # 注意：推理中 add_new_points_or_box 会做 normalize + scale to image_size
            # 训练时直接给像素坐标，已经在image_size空间下（因为已resize到768x768）
            point_inputs[key] = {
                'point_coords': coords.unsqueeze(0),   # (1, N, 2)
                'point_labels': labels.unsqueeze(0),    # (1, N)
            }

        elif key == 'pts':
            assert points_xyz is not None, "points_xyz required for pts modal"

            # 采样正点 3D 坐标
            if len(pos_indices) > 0 and num_pos > 0:
                sel_pos = pos_indices[torch.randperm(len(pos_indices), device=device)[:num_pos]]
                pos_coords = points_xyz[sel_pos]  # (num_pos, 3)
            else:
                pos_coords = torch.zeros(0, 3, device=device)

            if len(neg_indices) > 0 and num_neg > 0:
                sel_neg = neg_indices[torch.randperm(len(neg_indices), device=device)[:num_neg]]
                neg_coords = points_xyz[sel_neg]
            else:
                neg_coords = torch.zeros(0, 3, device=device)

            coords = torch.cat([pos_coords, neg_coords], dim=0)
            labels = torch.cat([
                torch.ones(pos_coords.shape[0], dtype=torch.int32, device=device),
                torch.zeros(neg_coords.shape[0], dtype=torch.int32, device=device),
            ])

            point_inputs[key] = {
                'point_coords': coords.unsqueeze(0),   # (1, N, 3)
                'point_labels': labels.unsqueeze(0),    # (1, N)
            }

    return point_inputs


def mask_to_box_with_jitter(gt_mask, points_xyz=None, jitter_ratio=0.1):
    """从GT mask生成 bounding box 提示（带随机扰动）
    
    输出格式与 sam4d_infer.py 中 box_prompts 一致:
    - img: 2D box → 2个角点 (左上, 右下), labels=[2, 3]
    - pts: 3D box → 2个角点 (min_xyz, max_xyz), labels=[2, 3]
    
    Args:
        gt_mask: dict of modal masks
        points_xyz: (N, 3) 点云坐标
        jitter_ratio: box边界抖动比例
    Returns:
        box_prompts: dict 格式同推理中的 box_prompts
    """
    box_prompts = {}

    for key, mask in gt_mask.items():
        device = mask.device

        if key == 'img':
            flat = mask.squeeze().bool()
            if flat.dim() == 1:
                raise ValueError("img mask should be 2D")
            H, W = flat.shape[-2], flat.shape[-1]
            ys, xs = torch.where(flat.squeeze())
            if len(ys) == 0:
                # fallback: 全图box
                box_prompts[key] = {
                    'point_coords': torch.tensor([[[0, 0], [W - 1, H - 1]]], dtype=torch.float32, device=device),
                    'point_labels': torch.tensor([[2, 3]], dtype=torch.int32, device=device),
                }
                continue

            x_min, x_max = xs.min().float(), xs.max().float()
            y_min, y_max = ys.min().float(), ys.max().float()

            # jitter
            w, h = x_max - x_min, y_max - y_min
            jx = (torch.rand(1, device=device) * 2 - 1) * w * jitter_ratio
            jy = (torch.rand(1, device=device) * 2 - 1) * h * jitter_ratio
            x_min = (x_min - jx.abs()).clamp(0, W - 1)
            y_min = (y_min - jy.abs()).clamp(0, H - 1)
            x_max = (x_max + jx.abs()).clamp(0, W - 1)
            y_max = (y_max + jy.abs()).clamp(0, H - 1)

            # 格式: (1, 2, 2)，与 add_new_points_or_box 中 box concat 到 point_coords 一致
            coords = torch.tensor([[[x_min.item(), y_min.item()],
                                     [x_max.item(), y_max.item()]]], device=device)
            labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
            box_prompts[key] = {'point_coords': coords, 'point_labels': labels}

        elif key == 'pts':
            assert points_xyz is not None
            flat = mask.squeeze().view(-1).bool()
            pos_idx = torch.where(flat)[0]
            if len(pos_idx) == 0:
                extent = points_xyz.max(0).values - points_xyz.min(0).values
                box_prompts[key] = {
                    'point_coords': torch.stack([points_xyz.min(0).values,
                                                  points_xyz.max(0).values]).unsqueeze(0),
                    'point_labels': torch.tensor([[2, 3]], dtype=torch.int32, device=device),
                }
                continue

            pos_pts = points_xyz[pos_idx]
            min_c = pos_pts.min(dim=0).values
            max_c = pos_pts.max(dim=0).values
            extent = max_c - min_c

            jitter = (torch.rand(3, device=device) * 2 - 1) * extent * jitter_ratio
            min_c = min_c - jitter.abs()
            max_c = max_c + jitter.abs()

            coords = torch.stack([min_c, max_c]).unsqueeze(0)  # (1, 2, 3)
            labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
            box_prompts[key] = {'point_coords': coords, 'point_labels': labels}

    return box_prompts


def compute_iou(pred_mask_logits, gt_mask):
    """计算预测mask与GT mask之间的IoU
    
    Args:
        pred_mask_logits: dict of {key: (B, 1, ...) logits}
        gt_mask: dict of {key: (B, 1, ...) binary}
    Returns:
        ious: dict of {key: (B, 1) float tensor}
    """
    ious = {}
    for key in pred_mask_logits:
        pred = (pred_mask_logits[key] > 0).float().view(pred_mask_logits[key].size(0), -1)
        gt = gt_mask[key].float().view(gt_mask[key].size(0), -1)
        intersection = (pred * gt).sum(dim=-1)
        union = pred.sum(dim=-1) + gt.sum(dim=-1) - intersection
        iou = intersection / (union + 1e-6)
        ious[key] = iou.unsqueeze(-1)  # (B, 1)
    return ious


def accumulate_loss(total_loss, new_loss):
    """累加loss字典
    
    Args:
        total_loss: dict of {loss_name: scalar tensor} or empty dict
        new_loss: dict of {loss_name: scalar tensor}
    Returns:
        updated total_loss dict
    """
    if not total_loss:
        return {k: v.clone() for k, v in new_loss.items()}
    for k, v in new_loss.items():
        if k in total_loss:
            total_loss[k] = total_loss[k] + v
        else:
            total_loss[k] = v.clone()
    return total_loss


def custom_collate(batch):
    """自定义collate函数，处理变长点云和多帧序列
    
    batch: list of dicts, 每个dict来自SAM4DTrainDataset.__getitem__
    由于训练时 batch_size=1（每GPU一个序列），实际上batch长度为1
    """
    assert len(batch) == 1, (
        f"SAM4D training expects batch_size=1 per GPU (each batch is a sequence "
        f"of 8 frames with up to 6 objects), but got batch_size={len(batch)}"
    )
    return batch[0]
