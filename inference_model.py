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
from preprocessing_data import OIDDataSet, KerasOIDDataset, form_classes_code_list, convert_yolobbox2bbox
from model_constructors.YOLOv4.utils import get_detection_data, draw_bbox

# Importing global variables from globals.py
from globals import CLASSES_NAME, CLASSES_PATH, TRAIN_DETECTION_BBOX, TRAIN_DETECTION_IMAGE_LABELS, IMAGE_SIZE


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

    dataset = KerasOIDDataset("train", classes_codes_list=classes_codes_list, class_description_df=class_description_df,
                              ground_truth_df=ground_truth_df, list_ids=list_ids)
    image_data, bboxes_data = dataset.__getitem__(17)

    tensor_image = image_data[17]
    image_shape = tensor_image.shape

    # image = tf.io.read_file(image_paths[25])
    # tensor_size = tf.constant(IMAGE_SIZE)
    # tensor_image = tf.image.decode_image(image, channels=3)
    # tensor_image = tf.image.convert_image_dtype(tensor_image, tf.float32)
    # tensor_image = tf.image.resize(tensor_image, tensor_size)
    #
    # # Define image height and width for denormalise the object coordinates
    image_height = tensor_image.shape[0]
    image_width = tensor_image.shape[1]
    print(image_height, image_width)

    target_image = tensor_image
    plt.figure(figsize=(5, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(target_image)
    plt.axis("off")

    # bboxes_df = ground_truth_df.loc[ground_truth_df["ImageID"] == list_ids[25]]
    # bboxes_df = bboxes_df.loc[bboxes_df["Label"].isin(classes_codes_list),
    #                           ["XMin", "YMin", "XMax", "YMax", "Label"]]
    #
    # # Correct bboxes label
    # replace_dict = {class_code: class_number for class_number, class_code in enumerate(classes_codes_list)}
    # bboxes_df["Label"] = bboxes_df.Label.replace(replace_dict).astype("int")
    #
    # # Define bboxes list
    # bboxes = bboxes_df.values.tolist()
    # print(bboxes)

    bboxes = bboxes_data

    # ToDo: Refactor code for testing data generator
    detections = get_detection_data(target_image, bboxes, CLASSES_NAME)

    # for bbox in bboxes:
    #
    #     # Denormalize object coordinates
    #     norm_center_x, norm_center_y = bbox[0] * image_width, bbox[1] * image_height
    #     norm_bbox_width, norm_bbox_height = bbox[2] * image_width, bbox[3] * image_height
    #
    #     x_min, y_min, x_max, y_max = convert_yolobbox2bbox(norm_center_x, norm_center_y, norm_bbox_width, norm_bbox_height)
    #
    #     # Define class name for this object
    #     # class_index = tf.argmax(bbox[0]).numpy()[0]
    #     class_index = int(bbox[-1])
    #     class_name = CLASSES_NAME[class_index]
    #     print(class_name)
    #
    #     # Draw rectangle for object detection with class name
    #     text_y_location = y_min - 10
    #     if text_y_location < 20:
    #         text_y_location = y_min + 20
    #     cv2.rectangle(target_image, pt1=(x_min, y_min), pt2=(x_max, y_max), color=(255, 0, 0), thickness=1)
    #     cv2.putText(target_image, class_name, (x_min, text_y_location), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(target_image)
    # plt.axis("off")
    # plt.show()
