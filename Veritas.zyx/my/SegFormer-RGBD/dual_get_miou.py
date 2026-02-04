import os

from PIL import Image
from tqdm import tqdm

from dual_segformer import SegFormer_Segmentation
from utils.utils_metrics import compute_mIoU, show_results


if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode is used to specify the content calculated when this file runs
    #   miou_mode=0 represents the entire miou calculation process, including obtaining prediction results and calculating miou.
    #   miou_mode=1 represents only obtaining prediction results.
    #   miou_mode=2 represents only calculating miou.
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    num_classes     = 16
    #--------------------------------------------#
    name_classes    = [
    "_background_",
    "person",
    "cable",
    "tube",
    "indicator",
    "electrical equipment",
    "electronic equipment",
    "mining equipment",
    "rail",
    "support equipment",
    "door",
    "tools and materials",
    "rescue equipment",
    "container",
    "metal fixture",
    "anchor",
    ]
    #-------------------------------------------------------#
    #   Points to the folder where the VOC dataset is located
    #   By default, it points to the VOC dataset in the root directory
    #-------------------------------------------------------#
    VOCdevkit_path  = 'datasets'

    image_ids       = open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val.txt"),'r').read().splitlines()
    depth_ids       = open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val.txt"),'r').read().splitlines()
    gt_dir          = os.path.join(VOCdevkit_path, "SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        segformer = SegFormer_Segmentation()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            depth_path  = os.path.join(VOCdevkit_path, "DepthImages/"+image_id+".png")
            depth       = Image.open(depth_path)
            image       = segformer.get_miou_png(image,depth)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
