
from sam4d.train.pipelines.compose import compose
from configs.sam4d_hieraS_mink34w32 import model as sam4d_model_cfg

class TrainConfig:
    data_root = 'D:/waymo_4d/val'
    work_dir = './work_dirs/sam4d'
    num_frames = 8
    max_num_objects = 6
    num_frames_to_correct_for_train = 2
    total_steps = 44000
    log_interval = 100
    save_interval = 1000
    num_workers = 4
    max_lr = 5e-5
    init_lr = 5e-6
    weight_decay = 0.1
    layer_wise_decay = 0.9
    sequence_aug_params = {
        'hflip_prob': 0.5,
        'affine_degrees': 25,
        'lidar_rot_z_degrees': 45,
        'lidar_hflip_prob': 0.5,
        'lidar_vflip_prob': 0.5,
    }
    frame_aug_params = {
        'brightness': 0.1,
        'contrast': 0.03,
        'saturation': 0.03,
        'grayscale_prob': 0.05,
    }
    # 定义pipeline（compose函数）
    train_pipeline = compose
    # 直接提供模型结构配置，供训练流程用
    backbone_cfg = sam4d_model_cfg['backbone']
    fusion_cfg = sam4d_model_cfg['fusion']
    head_cfg = sam4d_model_cfg['head']