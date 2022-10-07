# Importing TensorFlow & Keras libraries
import tensorflow as tf
from keras.utils import image_utils, dataset_utils

# Importing other libraries
import numpy as np
import pandas as pd
import os
import cv2

# Importing global variables
from globals import BATCH_SIZE, IMAGE_SIZE, ALLOW_IMAGE_FORMATS


class OIDDataSet:
    """"""

    def __init__(self, grond_truth_df, class_description_df, class_names,
                 label_mode="int", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
                 color_channels=3, random_seed=17, shuffle=True, interpolation="bilinear"):
        """"""
        self.image_paths = None
        self.base_path = None
        self.ground_truth_df = grond_truth_df
        self.class_description_df = class_description_df
        self.class_names = class_names
        self.label_mode = label_mode
        self.batch_size = batch_size
        self.image_size = image_size
        self.color_channels = color_channels
        self.random_seed = random_seed
        self.shuffle = shuffle

        # Defining "labels" variable
        self.labels = np.array([])

        self.interpolation = image_utils.get_interpolation(interpolation)

    def getting_dataset(self, subset="train"):
        """"""
        # Prepare base path
        if subset == "train":
            self.base_path = "dataset\\train"
        elif subset == "validation":
            self.base_path = "dataset\\validation"
        elif subset == "test":
            self.base_path = "dataset\\test"

        # Prepare image paths
        self.image_paths, _, _ = dataset_utils.index_directory(self.base_path, labels=None,
                                                               formats=ALLOW_IMAGE_FORMATS,
                                                               shuffle=self.shuffle, seed=self.random_seed)

        # Prepare ground truth
        for image_path in self.image_paths:
            image_id = image_path.split("\\")[-1].replace(".jpg", "")
            class_code_list = self.class_description_df.loc[self.class_description_df["LabelName"].isin(self.class_names),
                                                            "LabelCode"].tolist()

            image_ground_truth_df = self.ground_truth_df.loc[self.ground_truth_df["ImageID"] == image_id]
            image_ground_truth_df = image_ground_truth_df.loc[image_ground_truth_df["LabelName"].isin(class_code_list)]

            bboxes_index = image_ground_truth_df.index.tolist()

            # image = cv2.imread(image_path)
            # image_shape = image.shape()

            for bbox_index in bboxes_index:
                # Extract coordinates from image_ground_truth_df
                x_min, x_max = image_ground_truth_df.XMin[bbox_index], image_ground_truth_df.XMax[bbox_index]
                y_min, y_max = image_ground_truth_df.YMin[bbox_index], image_ground_truth_df.YMax[bbox_index]

                # Denormalize coordinates
                # x_min, x_max = int(x_min * image_shape[1]), int(x_max * image_shape[1])
                # y_min, y_max = int(y_min * image_shape[0]), int(y_max * image_shape[0])

                # Defining which class in inside this bounding box
                class_name = self.class_description_df.loc[self.class_description_df["LabelCode"] ==
                                                           image_ground_truth_df.LabelName[bbox_index],
                                                           "LabelName"].tolist()[0]
                print(class_name)

                classes_name_one_hot_df = pd.get_dummies(self.class_description_df,
                                                         columns=["LabelName"],
                                                         prefix="Label")
                print(classes_name_one_hot_df.head())

                class_name_one_hot = classes_name_one_hot_df.loc[classes_name_one_hot_df["LabelCode"] ==
                                                                 image_ground_truth_df.LabelName[bbox_index],
                                                                 "Label_Bear", "Label_Bird", "Label_Cat"].tolist()
                print(class_name_one_hot)

                # Making ground truth variable for one image
                result_ground_truth = (class_name_one_hot, (x_min, y_min, x_max, y_max))
                tensor_result = tf.constant(result_ground_truth)

                # Filling "labels" variable
                self.labels = np.append(self.labels, tensor_result)

        dataset = self.paths_and_labels_to_dataset(
            num_classes=len(self.class_names),
        )

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        if self.batch_size is not None:
            if self.shuffle:
                # Shuffle locally at each iteration
                dataset = dataset.shuffle(buffer_size=self.batch_size * 8, seed=self.random_seed)
            dataset = dataset.batch(self.batch_size)
        else:
            if self.shuffle:
                dataset = dataset.shuffle(buffer_size=1024, seed=self.random_seed)

        # Create reference "class_name"
        dataset.class_names = self.class_names
        # Include file paths for images as attribute
        dataset.file_paths = self.image_paths

        return dataset

    def paths_and_labels_to_dataset(self, num_classes):
        """"""
        paths_ds = tf.data.Dataset.from_tensor_slices(self.image_paths)
        image_ds = paths_ds.map(
            lambda x: self.load_image(x), num_parallel_calls=tf.data.AUTOTUNE
        )
        if self.label_mode:
            label_ds = dataset_utils.labels_to_dataset(self.labels, self.label_mode, num_classes)
            image_ds = tf.data.Dataset.zip((image_ds, label_ds))
        return image_ds

    def load_image(self, path):
        """"""
        image = tf.io.read_file(path)
        image = tf.image.decode_image(
            image, channels=self.color_channels, expand_animations=False
        )
        image = tf.image.resize(image, self.image_size, method=self.interpolation)
        image.set_shape((self.image_size[0], self.image_size[1], self.color_channels))
        return image

