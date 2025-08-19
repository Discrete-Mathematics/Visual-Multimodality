from .. import *

# Dataset config
"""Dataset Path"""
C.dataset_name = "MUSeg"
C.dataset_path = osp.join(C.root_dir, "MUSeg")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".jpg"
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.gt_format = ".png"
C.gt_transform = True
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = True  # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True
C.num_train_imgs = 1595
C.num_eval_imgs = 1576
C.num_classes = 15
C.class_names = [
    "person",
    "cable",
    "tube",
    "indicator",
    "electrical equipment",
    "electronic equipment",
    "mining equipment",
    "rail area",
    "support equipment",
    "door",
    "tools and materials",
    "rescue equipment",
    "container",
    "metal fixture",
    "anchoring equipment"
]

"""Image Config"""
C.background = 255
# 修改为640x640以更好地保留1082x932图片的信息
C.image_height = 640
C.image_width = 640
# 新增：是否使用resize而不是crop
C.use_resize_instead_of_crop = True
# 新增：resize的目标尺寸
C.resize_size = [640, 640]
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])
