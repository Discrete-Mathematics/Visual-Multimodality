import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from .train_model import SAM4DTrain
from .pipelines.loading import SAM4DTrainDataset
from .utils_train import custom_collate

def build_optimizer(model, cfg):
    """构建优化器 + layer-wise decay
    论文参数 (Table A1):
      - optimizer: AdamW
      - max_lr: 5e-5
      - init_lr: 5e-6
      - weight_decay: 0.1
      - layer_wise_decay: 0.9
    """
    param_groups = []

    # Backbone (image encoder) 使用 layer-wise decay
    num_layers = len(list(model.image_encoder.parameters()))
    for i, (name, param) in enumerate(model.image_encoder.named_parameters()):
        decay_rate = cfg.layer_wise_decay ** (num_layers - i)  # 论文: 0.9
        param_groups.append({
            'params': [param],
            'lr': cfg.max_lr * decay_rate,            # 论文: 5e-5 * decay
            'weight_decay': cfg.weight_decay,          # 论文: 0.1
        })

    # 其他模块正常学习率
    other_params = []
    for name, module in model.named_children():
        if name != 'image_encoder':
            other_params.extend(module.parameters())

    param_groups.append({
        'params': other_params,
        'lr': cfg.max_lr,                             # 论文: 5e-5
        'weight_decay': cfg.weight_decay,
    })

    optimizer = torch.optim.AdamW(param_groups,
                                   betas=(0.9, 0.999))  # 论文: (0.9, 0.999)
    return optimizer


def build_scheduler(optimizer, cfg):
    """构建学习率调度器
    论文参数 (Table A1):
      - scheduler: OneCycleLR
      - total_steps: 44000
      - warmup: 7500 steps
      - anneal: cosine
    """
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.max_lr,                    # 论文: 5e-5
        total_steps=cfg.total_steps,          # 论文: 44000
        pct_start=7500 / cfg.total_steps,     # 论文: warmup 7500 steps / 44000 ≈ 0.17
        anneal_strategy='cos',                # 论文: cosine
        div_factor=cfg.max_lr / cfg.init_lr,  # 论文: 5e-5 / 5e-6 = 10
    )
    return scheduler


def train(cfg):
    print("[train] data_root:", cfg.data_root)
    print("[train] work_dir:", cfg.work_dir)
    # ---- 分布式/单卡自动切换 ----
    use_ddp = 'LOCAL_RANK' in os.environ
    if use_ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        model = SAM4DTrain(cfg).cuda(local_rank)
        optimizer = build_optimizer(model, cfg)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        print("[train] 单卡模式 (无分布式环境变量)")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SAM4DTrain(cfg).to(device)
        optimizer = build_optimizer(model, cfg)

    # ---- 构建数据 ----
    dataset = SAM4DTrainDataset(
        data_root=cfg.data_root,
        ann_file=getattr(cfg, 'ann_file', None),
        pipeline=cfg.train_pipeline,
        num_frames=getattr(cfg, 'num_frames', 8),
        max_num_objects=getattr(cfg, 'max_num_objects', 6),
    )
    print("[train] dataset size:", len(dataset))
    if use_ddp:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=1,
        sampler=sampler,
        num_workers=getattr(cfg, 'num_workers', 4),
        collate_fn=custom_collate,
    )

    # ---- 优化器 & 调度器 ----
    scheduler = build_scheduler(optimizer, cfg)

    # ---- 混合精度 ----
    # 论文: bfloat16
    scaler = GradScaler()
    print("[train] 开始训练...")

    # ---- 训练循环 ----
    # 论文: total_steps = 44000, gradient_clip = 0.1
    global_step = 0
    while global_step < cfg.total_steps:  # 论文: 44000
        if use_ddp and sampler is not None:
            sampler.set_epoch(global_step)
        for batch in dataloader:
            if global_step >= cfg.total_steps:
                break

            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16):  # 论文: bfloat16
                losses = model(batch)
                total_loss = losses['total']

            scaler.scale(total_loss).backward()

            # 梯度裁剪 - 论文: max_norm=0.1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=0.1    # 论文: 0.1
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            if global_step % cfg.log_interval == 0:
                print(f"Step {global_step}/{cfg.total_steps}, "
                      f"Loss: {total_loss.item():.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")

            if global_step % cfg.save_interval == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, cfg)

    if use_ddp:
        dist.destroy_process_group()

def save_checkpoint(model, optimizer, scheduler, step, cfg):
    # Handle both DDP and non-DDP models
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    state = {
        'model': model_state,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
    }
    os.makedirs(cfg.work_dir, exist_ok=True)
    path = os.path.join(cfg.work_dir, f'checkpoint_{step}.pth')
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")

if __name__ == "__main__":
    from configs.configs_train import TrainConfig
    cfg = TrainConfig()
    train(cfg)