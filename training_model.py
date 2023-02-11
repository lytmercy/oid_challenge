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

# Importing global variables from globals.py
from globals import CLASSES_NAME, CLASSES_PATH, TRAIN_DETECTION_BBOX, TRAIN_DETECTION_IMAGE_LABELS, IMAGE_SIZE
from globals import EPOCHS, BATCH_SIZE

# Use
# matplotlib.use("TkAgg")


def train_model(model, model_type):
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

    train_data_gen = OIDDataset("train", classes_codes_list=classes_codes_list, class_description_df=class_description_df,
                                ground_truth_df=ground_truth_df, list_ids=list_ids, model_type=model_type)
    valid_data_gen = OIDDataset("validation", classes_codes_list=classes_codes_list, class_description_df=class_description_df,
                                ground_truth_df=ground_truth_df, list_ids=list_ids, model_type=model_type)

    model.yolo_fit(train_data_gen,
                   initial_epoch=0,
                   epochs=EPOCHS,
                   val_data_gen=valid_data_gen,
                   callbacks=[])

    if model_type == "yolo":
        model.save_model("entire_models\\YOLOv4\\")

    return model
