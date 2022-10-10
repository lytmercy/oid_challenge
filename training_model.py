# Importing TensorFlow & Keras libraries
import tensorflow as tf

# Importing other libraries
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import os

# Importing class for preprocess and getting images and labels (labels and coordinates of bbox)
from preprocessing_data import OIDDataSet

# Importing global variables from globals.py
from globals import CLASSES_NAME, CLASSES_PATH, TRAIN_DETECTION_BBOX, TRAIN_DETECTION_IMAGE_LABELS


def train_model():
    """"""

    class_description_df = pd.read_csv(CLASSES_PATH)
    class_description_df.columns = ["LabelCode", "LabelName"]
    ground_truth_df = pd.read_csv(TRAIN_DETECTION_BBOX, usecols=["ImageID", "LabelName",
                                                                 "XMin", "XMax", "YMin", "YMax"])

    image_ids = []
    for image_path in os.listdir("dataset\\train"):
        image_ids.append(image_path.split("\\")[-1].replace(".jpg", ""))

    ground_truth_df = ground_truth_df.loc[ground_truth_df["ImageID"].isin(image_ids)]

    preprocess_data = OIDDataSet(ground_truth_df, class_description_df, CLASSES_NAME)
    train_dataset = preprocess_data.getting_dataset()

    for images, labels in train_dataset.take(1):
        print(images[17])
        print(labels[17])

        # Define image height and width for denormalizing the object coordinates
        image_height = images[17].shape[0]
        image_width = images[17].shape[1]

        target_image = images[17].numpy()
        plt.imshow(target_image)
        plt.axis("off")
        plt.show()

        for image_labels in labels[17]:

            # Denormalize object coordinates
            norm_x_min, norm_x_max = image_labels[1][0].numpy(), image_labels[1][2].numpy()
            norm_y_min, norm_y_max = image_labels[1][1].numpy(), image_labels[1][3].numpy()

            x_min, x_max = int(norm_x_min * image_width), int(norm_x_min * image_width)
            y_min, y_max = int(norm_y_min * image_height), int(norm_y_max * image_height)

            # Define class name for this object
            class_index = tf.argmax(image_labels[0]).numpy()[0]
            print(image_labels[0].numpy())
            class_name = CLASSES_NAME[class_index]

            # Draw rectangle for object detection with class name
            cv2.rectangle(target_image, pt1=(x_min, y_min), pt2=(x_max, y_max), color=(255, 0, 0), thickness=2)
            cv2.putText(target_image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        plt.imshow(target_image)
        plt.axis("off")
        plt.show()
