import datetime
import os
from functools import partial
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from nets.segformer_dual import SegFormerRGBD
from utils.dual_dataloader import SegmentationDataset,seg_dataset_collate
from utils.dual_utils_fit import fit_one_epoch
from utils.dual_callbacks import EvalCallback, LossHistory
from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)

if __name__ == "__main__":
    # ---------------------------------#
    #   Cuda    Whether to use CUDA
    # ---------------------------------#
    Cuda = True
    # ----------------------------------------------#
    #   Seed    Used to fix random seed
    # ----------------------------------------------#
    seed = 11
    # ---------------------------------------------------------------------#
    #   distributed     Whether to use single-machine multi-GPU distributed training
    #                  Terminal commands only support Ubuntu. CUDA_VISIBLE_DEVICES is used to specify GPUs on Ubuntu.
    #                  Windows system defaults to using DP mode to utilize all GPUs, DDP is not supported.
    #   DP mode:
    #       Set            distributed = False
    #       Run in terminal:    CUDA_VISIBLE_DEVICES=0,1 python dual_train.py
    #   DDP mode:
    #       Set            distributed = True
    #       Run in terminal:    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 dual_train.py
    # ---------------------------------------------------------------------#
    distributed = False
    # ---------------------------------------------------------------------#
    #   sync_bn     Whether to use sync_bn, available in DDP multi-GPU mode
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #                Can reduce about half of GPU memory usage, requires pytorch 1.7.1+
    # ---------------------------------------------------------------------#
    fp16 = True
    # -----------------------------------------------------#
    #   num_classes
    # -----------------------------------------------------#
    num_classes = 16
    # -------------------------------------------------------------------#
    #   Backbone network being used:
    #   b0, b1, b2, b3, b4, b5
    # -------------------------------------------------------------------#
    phi = "b2"
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      Whether to use pretrained weights for the backbone network
    #                   These are backbone weights loaded during model construction.
    #                   If model_path is set, backbone weights don't need loading, pretrained becomes meaningless.
    #                   If model_path is not set and pretrained = True, only backbone weights are loaded for training.
    #                   If model_path is not set and pretrained = False, Freeze_Train = False, training starts from scratch without frozen backbone.
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = True
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   Weight file download instructions are in README, available via cloud storage.
    #   Pretrained weights are generic across different datasets as features are generic.
    #
    #   If training gets interrupted, set model_path to weight file in logs folder to resume training.
    #   Adjust freeze phase or unfreeze phase parameters below to maintain epoch continuity.
    #
    #   When model_path = '', no model weights are loaded.
    #
    #   Here we use full model weights loaded in train.py, pretrain doesn't affect this loading.
    #   To train from backbone pretrained weights: set model_path = '', pretrain = True (only loads backbone).
    #   To train from scratch: set model_path = '', pretrain = False, Freeze_Train = False.
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = "logs/best_epoch_weights.pth"
    # ------------------------------#
    #   Input image size
    # ------------------------------#
    input_shape = [640, 480]

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   Training has two phases: freeze phase and unfreeze phase.
    #   Freeze phase is designed for users with limited hardware resources.
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   Freeze phase training parameters
    #   Init_Epoch          Starting epoch for current training (can be > Freeze_Epoch)
    #   Freeze_Epoch        Number of epochs for frozen training
    #                       (Disabled when Freeze_Train=False)
    #   Freeze_batch_size   Batch size during frozen training
    #                       (Disabled when Freeze_Train=False)
    # ------------------------------------------------------------------#
    Init_Epoch = 30
    Freeze_Epoch = 0
    Freeze_batch_size = 0
    # ------------------------------------------------------------------#
    #   Unfreeze phase training parameters
    #   UnFreeze_Epoch      Total training epochs
    #   Unfreeze_batch_size Batch size after unfreezing
    # ------------------------------------------------------------------#
    UnFreeze_Epoch = 150
    Unfreeze_batch_size = 8
    # ------------------------------------------------------------------#
    #   Freeze_Train    Whether to perform freeze training
    # ------------------------------------------------------------------#
    Freeze_Train = False
    # ------------------------------------------------------------------#
    #   Other training params: learning rate, optimizer, LR scheduling
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         Maximum learning rate
    #                   Recommended settings:
    #                   Adam optimizer: Init_lr=1e-4
    #                   AdamW optimizer: Init_lr=1e-4
    #                   SGD not recommended for Transformer models
    #   Min_lr          Minimum learning rate, defaults to 0.01 of maximum LR
    # ------------------------------------------------------------------#
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  Optimizer type: adam, adamw, or sgd
    #   momentum        Momentum parameter for optimizer
    #   weight_decay    Weight decay to prevent overfitting
    #                   Note: adam may cause weight_decay issues, recommend 0 for adam
    # ------------------------------------------------------------------#
    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 1e-2
    # ------------------------------------------------------------------#
    #   lr_decay_type   Learning rate decay type: 'step' or 'cos'
    # ------------------------------------------------------------------#
    lr_decay_type = 'cos'
    # ------------------------------------------------------------------#
    #   save_period     Save weights every N epochs
    # ------------------------------------------------------------------#
    save_period = 20
    # ------------------------------------------------------------------#
    #   save_dir        Folder to save weights and log files
    # ------------------------------------------------------------------#
    save_dir = 'logs'
    # ------------------------------------------------------------------#
    #   eval_flag       Whether to evaluate during training (on validation set)
    #   eval_period     Evaluate every N epochs (not recommended too frequently)
    #                   Evaluation is time-consuming and slows down training
    #   Note: mAP here differs from get_map.py because:
    #   (1) This evaluates on validation set
    #   (2) Conservative evaluation params for faster evaluation
    # ------------------------------------------------------------------#
    eval_flag = False
    eval_period = 10

    # ------------------------------------------------------------------#
    #   VOCdevkit_path  Dataset path
    # ------------------------------------------------------------------#
    VOCdevkit_path = 'datasets'
    # ------------------------------------------------------------------#
    #   Recommended settings:
    #   Set to True when:
    #   - Few classes (several)
    #   - Many classes (dozens) with large batch_size (>10)
    #   Set to False when:
    #   - Many classes (dozens) with small batch_size (<10)
    # ------------------------------------------------------------------#
    dice_loss = False
    # ------------------------------------------------------------------#
    #   Whether to use focal loss to address class imbalance
    # ------------------------------------------------------------------#
    focal_loss = False
    # ------------------------------------------------------------------#
    #   Whether to assign different loss weights to different classes
    #   Default is balanced weights.
    #   If set, ensure numpy array format with length matching num_classes.
    #   Example:
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    # ------------------------------------------------------------------#
    cls_weights = np.ones([num_classes], np.float32)
    # ------------------------------------------------------------------#
    #   num_workers     Whether to use multi-threaded data loading
    #                   1 means disabled
    #                   Enabling speeds up data loading but uses more memory
    #                   Sometimes slower in keras implementation
    #                   Enable when I/O is bottleneck (GPU much faster than image loading)
    # ------------------------------------------------------------------#
    num_workers = 4

    seed_everything(seed)
    # ------------------------------------------------------#
    #   Set which GPUs to use
    # ------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #----------------------------------------------------#
    #   Download Pretrained weight
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)  
            dist.barrier()
        else:
            download_weights(phi)

    model   = SegFormerRGBD(num_classes=num_classes, phi=phi, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        #   Please refer to README for the weights file, download from Baidu Netdisk
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   Load according to the key of the pre-trained weight and the key of the model
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   Displays no matching key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    #----------------------#
    #   Record Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Multi-card sync Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   Multiple cards running in parallel
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    #   Read the txt corresponding to the data set
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes = num_classes, phi = phi, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )

        wanted_step = 1.5e4 if optimizer_type == "adamw" else 0.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))


    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
        lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        
        train_dataset   = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None

        #---------------------------------------#
        #   Begin train
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
                lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag   = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, \
                dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
