import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.segformer_dual import SegFormerRGBD
from utils.utils import cvtColor, preprocess_input, resize_image, show_config

class SegFormer_Segmentation(object):
    _defaults = {
        #-------------------------------------------------------------------#
        #   model_path points to the weight file in the logs folder
        #-------------------------------------------------------------------#
        "model_path"        : "logs/best_epoch_weights.pth",
        #----------------------------------------#
        "num_classes"       : 16,
        #----------------------------------------#
        #   The backbone network being used：
        #   b0、b1、b2、b3、b4、b5
        #----------------------------------------#
        "phi"               : "b2",
        #----------------------------------------#
        #   The size of the input image
        #----------------------------------------#
        "input_shape"       : [640, 480],
        #-------------------------------------------------#
        #   The mix_type parameter is used to control the visualization of the detection results.
        #
        #   When mix_type = 0, the original image is mixed with the generated image.
        #   When mix_type = 1, only the generated graph is retained.
        #   When mix_type = 2, only the background is removed, leaving only the target in the original image.
        #-------------------------------------------------#
        "mix_type"          : 0,
        #-------------------------------#
        #   Whether to use Cuda
        #-------------------------------#
        "cuda"              : True,
    }

    #---------------------------------------------------#
    #   Initialize SegFormer
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   set_color
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   get model
        #---------------------------------------------------#
        self.generate()
        
        show_config(**self._defaults)
                    
    #---------------------------------------------------#
    #   get all classes
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #-------------------------------#
        #   load model
        #-------------------------------#
        self.net    = SegFormerRGBD(num_classes=self.num_classes, phi=self.phi, pretrained=False)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def get_miou_png(self, image, depth):
        # ---------------------------------------------------------#
        #   Convert the image to RGB format to prevent errors when predicting grayscale images
        #   The code only supports prediction of RGB images, all other types will be converted to RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        depth = cvtColor(depth)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]

        # ---------------------------------------------------------#
        #   Add gray bars to the image for distortion-free resizing
        #   Alternatively, direct resizing can be used for recognition
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        depth_data, _, _ = resize_image(depth, (self.input_shape[1], self.input_shape[0]))

        # ---------------------------------------------------------#
        #   Add batch_size dimension
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        depth_data = np.expand_dims(np.transpose(preprocess_input(np.array(depth_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            depths = torch.from_numpy(depth_data)

            if self.cuda:
                images = images.cuda()
                depths = depths.cuda()

            # ---------------------------------------------------#
            #   Feed the images into the network for prediction
            # ---------------------------------------------------#
            pr = self.net(images, depths)[0]

            # ---------------------------------------------------#
            #   Extract the class of each pixel
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            # --------------------------------------#
            #   Crop out the gray bars
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            # ---------------------------------------------------#
            #   Resize the image
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

            # ---------------------------------------------------#
            #   Extract the class of each pixel
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image