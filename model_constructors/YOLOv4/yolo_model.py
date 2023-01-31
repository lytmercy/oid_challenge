## Code reference:
# https://github.com/taipingeric/yolo-v4-tf.keras/blob/73dfe97c00a03ebb7fab00a5a0549b958172482a/models.py

import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda
from keras.optimizers.optimizer_v2.adam import Adam
from keras.models import Model, load_model

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
    """Class of YOLOv4 model for build and entire forming this model."""
    def __init__(self, weight_path=None, config=yolo_config):
        """

        :param weight_path:
        :param config:
        """

        # assert config["image_size"][0] == config["image_size"][1], "not support yet"
        assert config["image_size"][0] % config["strides"][-1] == 0, "must be a multiple of last stride"
        self.yolo_model = None
        self.training_model = None
        self.inference_model = None
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
        # core yolo model
        input_layer = Input(self.image_size)
        yolov4_output = yolo_neck(input_layer, self.num_classes)
        self.yolo_model = Model(input_layer, yolov4_output)

        # Build training model
        y_true = [
            Input(name="input_2", shape=(52, 52, 3, (self.num_classes + 5))),  # label small boxes
            Input(name="input_3", shape=(26, 26, 3, (self.num_classes + 5))),  # label medium boxes
            Input(name="input_4", shape=(13, 13, 3, (self.num_classes + 5))),  # label large boxes
            Input(name="input_5", shape=(self.max_boxes, 4)),  # true bboxes
        ]
        loss_list = Lambda(yolo_loss, name="yolo_loss",
                           arguments={"num_classes": self.num_classes,
                                      "iou_loss_thresh": self.iou_loss_thresh,
                                      "anchors": self.anchors})([*self.yolo_model.output, *y_true])
        self.training_model = Model([self.yolo_model.input, *y_true], loss_list)

        # Build inference model
        yolov4_output = yolo_head(yolov4_output, self.num_classes, self.anchors, self.xy_scale)
        # output: [boxes, scores, classes, valid_detections]
        self.inference_model = Model(input_layer,
                                     non_max_suppression(yolov4_output, self.image_size, self.num_classes,
                                                         iou_threshold=self.config["iou_threshold"],
                                                         score_threshold=self.config["score_threshold"]))

        if load_pretrained and self.weight_path and self.weight_path.endswith(".weights"):
            if self.weight_path.endswith(".weights"):
                load_weights(self.yolo_model, self.weight_path)
                print(f"load from {self.weight_path}")
            elif self.weight_path.endwith(".h5"):
                self.training_model.load_weights(self.weight_path)
                print(f"load from {self.weight_path}")

        self.training_model.compile(optimizer=Adam(learning_rate=1e-3),
                                    loss={"yolo_loss": lambda y_true, y_pred: y_pred})

    def load_model(self, path):
        """

        :param path:
        :return:
        """
        self.yolo_model = load_model(path, compile=False)
        yolov4_output = yolo_head(self.yolo_model.output, self.num_classes, self.anchors, self.xy_scale)
        self.inference_model = Model(self.yolo_model.input,
                                     # [boxes, scores, classes, valid_detections]
                                     non_max_suppression(yolov4_output, self.image_size, self.num_classes))

    def save_model(self, path):
        """

        :param path:
        :return:
        """
        self.yolo_model.save(path)

    def preprocess_image(self, image):
        """

        :param image:
        :return:
        """
        image = cv2.resize(image, self.image_size[:2])
        image = image / 255.
        return image

    def yolo_fit(self, train_data_gen, epochs, val_data_gen=None, initial_epoch=0, callbacks=None):
        """

        :param train_data_gen:
        :param epochs:
        :param val_data_gen:
        :param initial_epoch:
        :param callbacks:
        :return:
        """
        self.training_model.fit(train_data_gen,
                                steps_per_epoch=len(train_data_gen),
                                validation_data=val_data_gen,
                                validation_steps=len(val_data_gen),
                                epochs=epochs,
                                callbacks=callbacks,
                                initial_epoch=initial_epoch)

    def predict_image(self, raw_image, random_color=True, plot_image=True, fig_size=(10, 10), show_text=True, return_output=False):
        """

        :param raw_image:
        :param random_color:
        :param plot_image:
        :param fig_size:
        :param show_text:
        :param return_output:
        :return:
        """
        print("image shape: ", raw_image.shape)
        image = self.preprocess_image(raw_image)
        images = np.expand_dims(image, axis=0)
        pred_output = self.inference_model.predict(images)
        detections = get_detection_data(image=raw_image,
                                        model_outputs=pred_output,
                                        classes_name=self.classes_name)

        output_image = draw_bbox(raw_image, detections, cmap=self.class_color, random_color=random_color,
                                 fig_size=fig_size, show_text=show_text, show_image=plot_image)

        if return_output:
            return output_image, detections
        else:
            return detections

    def predict(self, image_path, random_color=True, plot_image=True, fig_size=(10, 10), show_text=True):
        """

        :param image_path:
        :param random_color:
        :param plot_image:
        :param fig_size:
        :param show_text:
        :return:
        """
        raw_image = cv2.imread(image_path)[:, :, ::-1]
        return self.predict_image(raw_image, random_color, plot_image, fig_size, show_text)

