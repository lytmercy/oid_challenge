# Importing TensorFlow & Keras libraries
import tensorflow as tf
from keras.utils import image_utils, dataset_utils, Sequence

# Importing other libraries
import numpy as np
import pandas as pd
import cv2
import os

# Importing global variables
from globals import BATCH_SIZE, IMAGE_SIZE, ALLOW_IMAGE_FORMATS, MAX_BBOXES
from model_constructors.configs import yolo_config


class OIDDataSet:
    """Class for creating a dataset object with the method for processing data from different subsets."""

    def __init__(self, grond_truth_df, class_description_df, class_names,
                 label_mode="binary", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
                 color_channels=3, random_seed=17, shuffle=True, interpolation="bilinear"):
        """
        Init method for remaining all information about a dataset;
        :param grond_truth_df: pandas DataFrame that contains all names & boundary boxes of the image;
        :param class_description_df: pandas DataFrame that contains a list of all classes from the dataset;
        :param class_names: list of class names that will be use;
        :param label_mode: String describing the encoding of "labels". Options are:
                            - "binary" indicates that the labels (there can be only 2) are encoded as
                              'float32' scalars with values 0 or 1 (e.g. for 'binary_crossentropy').
                            - "categorical" means that the labels are mapped into a categorical vector.
                              (e.g. for 'categorical_crossentropy' loss).
        :param batch_size: int size of batch for Dataset;
        :param image_size: tuples with 2 int numbers that be width and height of an image;
        :param color_channels: int number of colour encoding channels in an image;
        :param random_seed: int number of random state for getting same result from dataset everytime;
        :param shuffle: boolean variable that says shuffle this dataset or not;
        :param interpolation: string with the name of the interpolation method for resizing an image.
        """
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
        self.labels = np.array([[]])

        self.interpolation = image_utils.get_interpolation(interpolation)

    def getting_dataset(self, subset="train"):
        """"""
        # Prepare base path
        assert subset == "train" or subset == "validation" or subset == "test", "Not valid subset name! Please check subset name."
        self.base_path = f"dataset\\{subset}"

        # Prepare image paths
        self.image_paths, _, _ = dataset_utils.index_directory(self.base_path, labels=None,
                                                               formats=ALLOW_IMAGE_FORMATS,
                                                               shuffle=self.shuffle, seed=self.random_seed)

        # Prepare class description dataframe
        self.class_description_df = self.class_description_df.loc[
            self.class_description_df["LabelName"].isin(self.class_names)
        ]
        # Prepare class code list
        class_code_list = self.class_description_df.loc[:, "LabelCode"].tolist()

        # Prepare ground truth
        for image_path in self.image_paths:

            image_id = image_path.split("\\")[-1].split(".", 1)[0]

            image_ground_truth_df = self.ground_truth_df.loc[self.ground_truth_df["ImageID"] == image_id]
            image_ground_truth_df = image_ground_truth_df.loc[image_ground_truth_df["LabelName"].isin(class_code_list)]

            bboxes_index = image_ground_truth_df.index.tolist()

            bboxes_coordinates = np.zeros(shape=(MAX_BBOXES, 2, 4), dtype=np.float32)
            bbox_count = 0

            # image = cv2.imread(image_path)
            # image_shape = image.shape()

            for bbox_index in bboxes_index:
                if bbox_count == MAX_BBOXES:
                    break
                else:
                    # Extract coordinates from image_ground_truth_df
                    x_min, x_max = image_ground_truth_df.XMin[bbox_index], image_ground_truth_df.XMax[bbox_index]
                    y_min, y_max = image_ground_truth_df.YMin[bbox_index], image_ground_truth_df.YMax[bbox_index]

                    # Denormalize coordinates
                    # x_min, x_max = int(x_min * image_shape[1]), int(x_max * image_shape[1])
                    # y_min, y_max = int(y_min * image_shape[0]), int(y_max * image_shape[0])

                    # Defining which class in inside this bounding box
                    # class_name = self.class_description_df.loc[self.class_description_df["LabelCode"] ==
                    #                                            image_ground_truth_df.LabelName[bbox_index],
                    #                                            "LabelName"].tolist()[0]

                    # ToDo: Maybe have a rationale to make labels in numbers (from 0 to n)
                    classes_name_one_hot_df = pd.get_dummies(self.class_description_df,
                                                             columns=["LabelName"],
                                                             prefix="Label")

                    classes_name_one_hot = classes_name_one_hot_df.loc[classes_name_one_hot_df["LabelCode"] ==
                                                                     image_ground_truth_df.LabelName[bbox_index],
                                                                     ["Label_Bear", "Label_Bird", "Label_Cat"]]

                    classes_name_one_hot = classes_name_one_hot.iloc[0, :].tolist()
                    classes_name_one_hot.append(0)
                    # Define coordinates bbox
                    # ToDo: Change xy minmax system to yolo system (cx, cy, w, h)
                    bbox_coordinates = [x_min, y_min, x_max, y_max]

                    # Making ground truth variable for one image
                    labeled_bbox = tf.constant([classes_name_one_hot, bbox_coordinates], dtype=tf.float32)

                    # Filling numpy array with labeled bbox
                    bboxes_coordinates[bbox_count] = labeled_bbox
                    bbox_count += 1

            # Filling "labels" variable
            if self.labels.size == 0:
                self.labels = np.array([bboxes_coordinates])
            else:
                self.labels = np.append(self.labels, [bboxes_coordinates], axis=0)

        dataset = self.paths_and_labels_to_dataset(
            num_classes=len(self.class_names)
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
            self.labels = tf.constant(self.labels)
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
        return image/255.


def form_classes_code_list(class_description_df, class_names):
    """
    Forming pandas.DataFrame with classes codes and classes what they mean (classes name);
    :param class_description_df :type pandas.DataFrame: That contains a list of all classes from the dataset and their codes;
    :param class_names :type list: of class names that will be use in this project;
    :return: formed pandas.DataFrame with classes codes and classes what they mean that will be use in this project;
    """

    # Prepare class description dataframe
    class_description_df = class_description_df.loc[
        class_description_df["LabelName"].isin(class_names)
    ]
    # Prepare class code list
    classes_codes_list = class_description_df.loc[:, "LabelCode"].tolist()

    return classes_codes_list

class KerasOIDDataset(Sequence):
    """"""
    def __init__(self, subset, list_ids, grond_truth_df, classes_codes_df, annotation_lines,
                 label_mode="binary", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, max_boxes=MAX_BBOXES,
                 color_channels=3, random_seed=17, shuffle=True, interpolation="bilinear"):
        """
        Init method for remaining all information about a dataset;
        :param grond_truth_df: pandas DataFrame that contains all names & boundary boxes of the image;
        :param classes_codes_df :type pandas.DataFrame: that contains a list of needed classes for this project;
        :param label_mode: String describing the encoding of "labels". Options are:
                            - "binary" indicates that the labels (there can be only 2) are encoded as
                              'float32' scalars with values 0 or 1 (e.g. for 'binary_crossentropy').
                            - "categorical" means that the labels are mapped into a categorical vector.
                              (e.g. for 'categorical_crossentropy' loss).
        :param batch_size: int size of batch for Dataset;
        :param image_size: tuples with 2 int numbers that be width and height of an image;
        :param color_channels: int number of colour encoding channels in an image;
        :param random_seed: int number of random state for getting same result from dataset everytime;
        :param shuffle: boolean variable that says shuffle this dataset or not;
        :param interpolation: string with the name of the interpolation method for resizing an image.
        """

        self.base_path = f"dataset\\{subset}"
        self.list_ids = list_ids
        self.ground_truth_df = grond_truth_df
        self.classes_codes_list = classes_codes_df
        self.annotation_lines = annotation_lines
        self.num_classes = len(classes_codes_df)
        self.label_mode = label_mode
        self.batch_size = batch_size
        self.image_size = image_size
        self.color_channels = color_channels
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.num_gpu = 1
        self.anchors = np.arange(yolo_config["anchors"]).reshape((9, 2))
        self.num_indexes = np.arange(len(self.annotation_lines))
        self.max_boxes = max_boxes

        # init indexes for store all indexes
        # self.labels = np.array([[]])
        self.indexes = None

        self.interpolation = image_utils.get_interpolation(interpolation)
        self.on_epoch_end()

    def __len__(self):
        """"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """"""

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

        # Find list of IDs
        list_ids_batch = [self.list_ids[k] for k in indexes]

        # Generate X

    def on_epoch_end(self):
        """"""

        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle is True:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.indexes)


    def __data_generation(self):
        """"""



