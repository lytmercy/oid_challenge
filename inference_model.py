# Importing TensorFlow & Keras libraries
import tensorflow as tf

# Importing other libraries
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# Importing class for preprocess and getting images and labels (labels and coordinates of bbox)
from preprocessing_data import OIDDataset, form_classes_code_list
from model_constructors.YOLOv4.utils import get_detection_data, draw_bbox, xywh_to_x0y0x1y1
from model_constructors.YOLOv4.custom_layers import get_boxes

# Importing global variables from globals.py
from globals import CLASSES_NAME, CLASSES_PATH, TRAIN_DETECTION_BBOX, TRAIN_DETECTION_IMAGE_LABELS, IMAGE_SIZE
from model_constructors.configs import yolo_config


def getting_inference():
    """"""

    classes_codes_list, class_description_df = form_classes_code_list(CLASSES_PATH, CLASSES_NAME)

    ground_truth_df = pd.read_csv(TRAIN_DETECTION_BBOX, usecols=["ImageID", "LabelName",
                                                                 "XMin", "XMax", "YMin", "YMax"])
    ground_truth_df.rename(columns={"LabelName": "Label"}, inplace=True)

    image_paths = []
    list_ids = []
    for image_path in os.listdir("dataset\\train"):
        image_paths.append(os.path.join("dataset\\train\\", image_path))
        list_ids.append(image_path.split(".", 1)[0])

    dataset = OIDDataset("train", classes_codes_list=classes_codes_list, class_description_df=class_description_df,
                              ground_truth_df=ground_truth_df, list_ids=list_ids, model_type="yolo")
    dataset_outputs = dataset.__getitem__(17)

    tensor_image = dataset_outputs[0][0][5]
    image_shape = tensor_image.shape

    # Define image height and width for denormalise the object coordinates
    image_height = image_shape[0]
    image_width = image_shape[1]
    print(image_height, image_width)

    target_image = (tensor_image * 255.).astype("uint8")
    plt.figure(figsize=(5, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(target_image)
    plt.axis("off")
