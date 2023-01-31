import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda
from keras.optimizers.optimizer_v2.adam import Adam
from keras.models import Model

# Importing other libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json


# Import yolo libraries
from model_constructors.configs import yolo_config
from model_constructors.YOLOv4.custom_losses import yolo_loss
from model_constructors.YOLOv4.custom_layers import yolo_neck, yolo_head, non_max_suppression
from model_constructors.YOLOv4.utils import load_weights, get_detection_data, draw_bbox
from model_constructors.YOLOv4.utils import voc_ap, draw_plot_func, read_txt_to_list

# Import global variables
from globals import CLASSES_NAME


class YOLOv4(object):
    """Class of YOLOv4 model_weights for build and entire forming this model_weights."""
    def __init__(self, weight_path=None, config=yolo_config):
        """

        :param weight_path:
        :param config:
        """

        # assert config["image_size"][0] == config["image_size"][1], "not support yet"
        assert config["image_size"][0] % config["strides"][-1] == 0, "must be a multiple of last stride"
        self.classes_name = CLASSES_NAME
        self.image_size = config["image_size"]
        self.num_classes = len(self.classes_name)
        self.weight_path = weight_path
        self.anchors = np.array(config["anchors"]).reshape((3, 3, 2))
        self.xy_scale = config["xy_scale"]
        self.strides = config["strides"]
        self.output_sizes = [self.image_size[0] // s for s in self.strides]
        self.class_color = {name: list(np.random.random(size=3) * 255) for name in self.classes_name}

        # Training
        self.max_boxes = config["max_boxes"]
        self.iou_loss_thresh = config["iou_loss_thresh"]
        self.config = config
        assert self.num_classes < 0, "no classes detected!"

        K.clear_session()
        if self.config["num_gpu"] > 1:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                self.build_yolo(load_pretrained=True if self.weight_path else False)
        else:
            self.build_yolo(load_pretrained=True if self.weight_path else False)


    def build_yolo(self, load_pretrained=True):
        """
        Method for entire forming of YOLO model_weights with all components (backbone, neck and heads);
        :param load_pretrained:
        :return:
        """




