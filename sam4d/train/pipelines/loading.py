import torch
from torch.utils.data import Dataset
import os
import pickle
import json
import random
from ..utils_train import filter_and_sample_objects

class SAM4DTrainDataset(Dataset):
    def __init__(self, data_root, ann_file, pipeline,
                 num_frames=8, max_num_objects=6,
                 min_score=0.3, min_volume=10):
        self.data_root = data_root
        self.num_frames = num_frames
        self.max_num_objects = max_num_objects
        self.min_score = min_score
        self.min_volume = min_volume
        self.pipeline = pipeline
        self.scenes = self._load_annotations(ann_file)

    def _load_annotations(self, ann_file):
        """解析Waymo数据集目录结构，构建场景-帧索引"""
        scenes = []
        meta_dir = os.path.join(self.data_root, 'meta_infos')
        pcd_dir = os.path.join(self.data_root, 'pcds')
        label_dir = os.path.join(self.data_root, 'sam4d_labels')
        image_dir = os.path.join(self.data_root, 'undistort_images')

        # 获取所有segment名称（由.pkl文件标识）
        seg_names = [f.replace('.pkl', '') for f in os.listdir(meta_dir) if f.endswith('.pkl')]
        for seg_name in seg_names:
            # 加载元数据（时间戳、位姿、相机参数等）
            with open(os.path.join(meta_dir, seg_name + '.pkl'), 'rb') as f:
                meta = pickle.load(f)

            seg_pcd = os.path.join(pcd_dir, seg_name)
            seg_label = os.path.join(label_dir, seg_name)
            seg_img = os.path.join(image_dir, seg_name)

            if not os.path.exists(seg_pcd):
                continue

            # 获取该segment的所有帧ID（按数字排序）
            frame_ids = sorted([f.replace('.npz', '') for f in os.listdir(seg_pcd) if f.endswith('.npz')])
            frames_info = []
            for fid in frame_ids:
                frame_info = {
                    'segment': seg_name,
                    'frame_id': fid,
                    'timestamp': meta['timestamps'][int(fid)] if 'timestamps' in meta else None,
                    'pose': meta['poses'][int(fid)] if 'poses' in meta else None,
                    'camera_params': meta.get('camera_params', {}),
                    'pcd_path': os.path.join(seg_pcd, f'{fid}.npz'),
                    'label_path': os.path.join(seg_label, f'{fid}.json'),
                    'image_paths': {
                        cam: os.path.join(seg_img, fid, cam + '.jpg')
                        for cam in ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
                    }
                }
                frames_info.append(frame_info)

            if frames_info:
                scenes.append({'scene_id': seg_name, 'frames': frames_info})
        return scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        total_frames = len(scene['frames'])
        start = random.randint(0, max(0, total_frames - self.num_frames))
        sampled_frames = scene['frames'][start:start + self.num_frames]

        processed = []
        for frame_info in sampled_frames:
            data = self.pipeline(frame_info)   # 需实现加载图像、点云、标注
            processed.append(data)

        # 过滤目标（需保证每个目标包含跨帧序列）
        objects = filter_and_sample_objects(
            processed[0]['gt_instances'],
            max_num_objects=self.max_num_objects,
            min_score=self.min_score,
            min_volume=self.min_volume
        )
        return {
            'frames': processed,
            'objects': objects,
            'ego_poses': [f.get('ego_pose', None) for f in sampled_frames]
        }