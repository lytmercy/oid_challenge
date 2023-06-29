# Code reference:
# https://github.com/taipingeric/yolo-v4-tf.keras/blob/73dfe97c00a03ebb7fab00a5a0549b958172482a/models.py

import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda
from keras.optimizers.optimizer_v2.adam import Adam
from keras.models import Model, load_model

# Importing other libraries
import numpy as np
import cv2

# Import yolo libraries
from src.models.YOLOv4.custom_losses import yolo_loss
from src.models.YOLOv4.custom_layers import yolo_neck, yolo_head, non_max_suppression
from src.models.YOLOv4.utils import load_weights, get_detection_data, draw_bbox


class YOLOv4(object):
    """Class of YOLOv4 model for build and entire forming this model."""
    def __init__(self, config, model_config, classes_name, weight_path=None):
        """
        Initialize yolo config variables, weight path and yolo models (default, training  and inference);
        :param config: dict config with variable for yolo model;
        :param weight_path: path to file with weight for yolo model.
        """

        # assert config["image_size"][0] == config["image_size"][1], "not support yet"
        assert model_config.image_size[0] % model_config.strides[-1] == 0, "must be a multiple of last stride"
        self.classes_name = classes_name
        self.image_size = model_config.image_size
        self.color_channels = config.preprocess.color_channels
        self.num_classes = len(self.classes_name)
        self.weight_path = weight_path
        self.anchors = np.array(model_config.anchors).reshape((3, 3, 2))
        self.xy_scale = model_config.xy_scale
        self.strides = model_config.strides
        self.output_sizes = [self.image_size[0] // s for s in self.strides]
        self.class_color = {name: list(np.random.random(size=3) * 255) for name in self.classes_name}
        self.model_config = model_config
        self.config = config

        # Define model types
        self.yolo_model = None
        self.training_model = None
        self.inference_model = None

        # Training
        self.max_boxes = model_config.max_boxes
        self.iou_loss_thresh = model_config.iou_loss_thresh
        assert self.num_classes > 0, "no classes detected!"

        K.clear_session()
        if self.config.num_gpu > 1:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                self.build_yolo(load_pretrained=True if self.weight_path else False)
        else:
            self.build_yolo(load_pretrained=True if self.weight_path else False)

    def build_yolo(self, load_pretrained=True):
        """
        Entire forming of the YOLO model with all components (backbone, neck and heads);
        :param load_pretrained: if True then model will load weights from file;
        :returns: formed yolo default, training and inference model.
        """

        # core yolo model
        input_layer = Input(self.image_size + (self.color_channels,))
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
                                                         iou_threshold=self.model_config.iou_threshold,
                                                         score_threshold=self.model_config.score_threshold))

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
        Load yolo model for default and inference model with Keras function keras.models.load_model;
        :param path: path to file with model in ".h5" format ar model weight;
        :return: yolo default and inference model.
        """

        self.yolo_model = load_model(path, compile=False)
        yolov4_output = yolo_head(self.yolo_model.output, self.num_classes, self.anchors, self.xy_scale)
        self.inference_model = Model(self.yolo_model.input,
                                     # [boxes, scores, classes, valid_detections]
                                     non_max_suppression(yolov4_output, self.image_size, self.num_classes))

    def save_model(self, path):
        """
        Saving yolo model with keras method ".save()" for class keras.models.Model().
        :param path: path to file where save model in ".h5" format;
        """

        self.yolo_model.save(path)

    def preprocess_image(self, image):
        """
        Preprocess image for yolo model;
        :param image: sample raw image from dataset;
        :return: preprocessed image.
        """

        image = cv2.resize(image, self.image_size[:2])
        image = image / 255.
        return image

    def yolo_fit(self, train_data_gen, epochs, val_data_gen=None, initial_epoch=0, callbacks=None):
        """
        Fit yolo model with keras method ".fit()" for class keras.models.Model().
        :param train_data_gen: training data generator;
        :param epochs :type int: number of epochs for fitting process;
        :param val_data_gen: validation data generator;
        :param initial_epoch :type int: initial epoch for fitting process;
        :param callbacks :type list: of callbacks for fitting process;
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
        Predicting image with inference yolo model;
        :param raw_image: sample raw image from dataset;
        :param random_color: if True -- color will be random in draw_bbox() function;
        :param plot_image: if True -- image will be shown in plot after detection;
        :param fig_size: size of figure in plot;
        :param show_text: if True -- text will be shown over bboxes in plot;
        :param return_output: if True -- function return output image after prediction (with bboxes);
        :returns: detections data (and if return_output==True -- return image with detections data).
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
        Predict bboxes and label for image that given;
        :param image_path: path to image that will be use in prediction;
        :param random_color: if True -- color will be random in predict_image() method;
        :param plot_image: if True -- image will be shown in plot after prediction;
        :param fig_size: size of figure in plot;
        :param show_text: if True -- text will be shown over bboxes in plot;
        :return: prediction result from predict_image() method.
        """

        raw_image = cv2.imread(image_path)[:, :, ::-1]
        return self.predict_image(raw_image, random_color, plot_image, fig_size, show_text)

