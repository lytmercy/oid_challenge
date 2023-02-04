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


def form_classes_code_list(class_description_path, class_names):
    """
    Forming pandas.DataFrame with classes codes and classes what they mean (classes name);
    :param class_description_path :type string: path to file with class description data;
    :param class_names :type list: of class names that will be use in this project;
    :return: formed pandas.DataFrame with classes codes and classes what they mean that will be use in this project;
    """

    class_description_df = pd.read_csv(class_description_path)
    class_description_df.columns = ["LabelCode", "LabelName"]

    # Prepare class description dataframe
    class_description_df = class_description_df.loc[
        class_description_df["LabelName"].isin(class_names)
    ]

    # Prepare class code list
    classes_codes_list = class_description_df.loc[:, "LabelCode"].tolist()

    return classes_codes_list, class_description_df


def load_prepare_image(image_path, image_size):
    """"""

    image = tf.io.read_file(image_path)
    tensor_image_size = tf.constant(image_size)
    decoded_image = tf.image.decode_image(image, channels=3)  # colour images
    # Convert uint8 tensor to floats in the [0, 1] range
    decoded_image = tf.image.convert_image_dtype(decoded_image, tf.float32)
    # Resize the image into image_size
    decoded_image = tf.image.resize(decoded_image, size=tensor_image_size)

    return decoded_image


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """
    Preprocess true boxes to training input format;
    :param true_boxes :type array: shape=(batch_size, max_boxes_per_image, 5);
    Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape;
    :param input_shape :type list: height & width of image;
    :param anchors :type array: anchors for yolo model (shape=(N, 2), (9, width height));
    :param num_classes :type int: number of classes from dataset;
    :returns: y_true: list of array;
              shape: like yolo_outputs;
              xy & width height (wh): relative bbox value for yolo model.
    """
    num_stages = 3  # default setting for yolo, tiny yolo will be 2
    anchor_mask = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    bbox_per_grid = 3
    true_boxes = np.array(true_boxes, dtype="float32")
    true_boxes_abs = np.array(true_boxes, dtype="float32")
    input_shape = np.array(input_shape, dtype="int32")
    true_boxes_xy = (true_boxes_abs[..., 0:2] + true_boxes_abs[..., 2:4]) // 2  # (10, 2)
    true_boxes_wh = true_boxes_abs[..., 2:4] - true_boxes_abs[..., 0:2]  # (10, 2)

    # Normalize x,y,w,h relative to image size -> (0~1)
    true_boxes[..., 0:2] = true_boxes_xy/input_shape[::-1]  # xy
    true_boxes[..., 2:4] = true_boxes_wh/input_shape[::-1]  # wh

    batch_size = true_boxes.shape[0]
    grid_sizes = [input_shape//{0:8, 1:16, 2:32}[stage] for stage in range(num_stages)]
    y_true = [np.zeros((batch_size,
                        grid_sizes[stage][0],
                        grid_sizes[stage][1],
                        bbox_per_grid,
                        5+num_classes), dtype="float32")
              for stage in range(num_stages)]

    # [(?, 52, 52, 3, 5 + num_classes) (?, 26, 26, 3, 5 + num_classes) (?, 13, 13, 3, 5 + num_classes)]
    y_true_boxes_xywh = np.concatenate((true_boxes_xy, true_boxes_wh), axis=-1)
    # Expand dim to apply broadcasting
    anchors = np.expand_dims(anchors, 0)  # (1, 9, 2)
    anchor_maxes = anchors / 2.  # (1, 9, 2)
    anchor_mins = -anchor_maxes  # (1, 9, 2)
    valid_mask = true_boxes_wh[..., 0] > 0  # (1, 10)

    for batch_idx in range(batch_size):
        # Discard zero rows
        width_height = true_boxes_wh[batch_idx, valid_mask[batch_idx]]  # (# of bbox, 2)
        num_boxes = len(width_height)
        if num_boxes == 0: continue
        width_height = np.expand_dims(width_height, axis=-2)  # (# of bbox, 1, 2)
        box_maxes = width_height / 2.  # (# of bbox, 1, 2)
        box_mins = -box_maxes  # (# of bbox, 1, 2)

        # Compute IoU between each anchor and true boxes for responsibility assignment
        intersect_mins = np.maximum(box_mins, anchor_mins)  # (# of bbox, 9, 2)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = np.prod(intersect_wh, axis=-1)  # (9,)
        box_area = width_height[..., 0] * width_height[..., 1]  # (# of bbox, 1)
        anchor_area = anchors[..., 0] * anchors[..., 1]  # (1, 9)
        iou = intersect_area / (box_area + anchor_area - intersect_area)  # (# of bbox, 9)

        # Find best anchor for each true box
        best_anchors = np.argmax(iou, axis=-1)  # (# of bbox,)
        for box_idx in range(num_boxes):
            best_anchor = best_anchors[box_idx]
            for stage in range(num_stages):
                if best_anchor in anchor_mask[stage]:
                    x_offset = true_boxes[batch_idx, box_idx, 0] * grid_sizes[stage][1]
                    y_offset = true_boxes[batch_idx, box_idx, 1] * grid_sizes[stage][0]
                    # Grid Index
                    grid_col = np.floor(x_offset).astype("int32")
                    grid_row = np.floor(y_offset).astype("int32")
                    anchor_idx = anchor_mask[stage].index(best_anchor)
                    class_idx = true_boxes[batch_idx, box_idx, 4].astype("int32")
                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, :2] = true_boxes_xy[batch_idx, box_idx, :]  # abs xy
                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 2:4] = true_boxes_xy[batch_idx, box_idx, :]  # abs wh
                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 4] = 1 # confidence

                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 5 + class_idx] = 1  # one-hot encoding

    return y_true, y_true_boxes_xywh


def convert_yolobbox2bbox(center_x, center_y, width, height):
    """
    Convert yolo bbox coordinate to x1,y1,x2,y2 bbox coordinate;
    :param center_x: center in x-axis of bbox in yolo coordinate;
    :param center_y: center in y-axis of bbox in yolo coordinate;
    :param width: width of bbox;
    :param height: height of bbox;
    :returns: coordinates in x1,y1 x2,y2 format.
    """
    x1, y1 = center_x - width/2, center_y - height/2
    x2, y2 = center_x + width/2, center_y + height/2

    return x1, y1, x2, y2

class KerasOIDDataset(Sequence):
    """"""
    def __init__(self, subset, list_ids, ground_truth_df, classes_codes_list, class_description_df, mode="fit",
                 label_mode="binary", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, max_boxes=MAX_BBOXES,
                 color_channels=3, random_seed=17, shuffle=True, interpolation="bilinear"):
        """
        Init method for remaining all information about a dataset;
        :param ground_truth_df: pandas DataFrame that contains all names & boundary boxes of the image;
        :param classes_codes_list :type pandas.DataFrame: that contains a list of needed classes for this project;
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
        self.ground_truth_df = ground_truth_df
        self.classes_codes_list = classes_codes_list
        self.class_description_df = class_description_df
        self.num_classes = len(classes_codes_list)
        self.label_mode = label_mode
        self.batch_size = batch_size
        self.image_size = image_size
        self.color_channels = color_channels
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.mode = mode
        self.num_gpu = 1
        self.anchors = np.array(yolo_config["anchors"]).reshape((9, 2))
        self.max_boxes = max_boxes

        # init indexes for store all indexes for loading to batch
        self.indexes = None

        self.image_paths = []
        for image_path in os.listdir(self.base_path):
            self.image_paths.append(os.path.join(f"{self.base_path}\\", image_path))

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

        # Generate X that contains images
        X, *y = self.__data_generation(list_ids_batch)

        # Check string attribute mode for "fit" mode
        if self.mode == "fit":
            # Return X (images) and y (ground truth -> boundary box)
            return X, y
        # Check string attribute mode for "predict" model
        elif self.mode == "predict":
            # Return only one X (images)
            return X
        else:
            # When mode is not "fit" or "predict" then rise Attribute Error
            raise AttributeError("The mode parameter should be set to 'fit' or 'predict'.")

    def on_epoch_end(self):
        """"""

        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle is True:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_ids_batch):
        """"""

        X = np.empty((len(list_ids_batch), *self.image_size +(self.color_channels,)), dtype=np.float32)

        y_bbox = np.empty((len(list_ids_batch), self.max_boxes, 5), dtype=np.float32)  # x1y1x2y2

        for i, image_id in enumerate(list_ids_batch):
            image_data, bbox_data = self.get_data(image_id, self.image_paths[i])
            X[i] = image_data
            y_bbox[i] = bbox_data

        y_tensor, y_true_boxes_xywh = preprocess_true_boxes(y_bbox, self.image_size, self.anchors, self.num_classes)

        return X, y_tensor, y_true_boxes_xywh

    def get_data(self, image_id, image_path):
        """"""

        height, width = self.image_size
        image_data = load_prepare_image(image_path, self.image_size)

        bboxes_df = self.ground_truth_df.loc[self.ground_truth_df["ImageID"] == image_id]
        bboxes_df = bboxes_df.loc[bboxes_df["Label"].isin(self.classes_codes_list),
                                  ["XMin", "YMin", "XMax", "YMax", "Label"]]

        # Correct bboxes label
        replace_dict = {class_code: class_number for class_number, class_code in enumerate(self.classes_codes_list)}
        bboxes_df["Label"] = bboxes_df.Label.replace(replace_dict)

        # Define bboxes list
        bboxes_list = bboxes_df.values.tolist()

        boxes = np.array(bboxes_list)

        # Correct bboxes coordinates
        bbox_data = np.zeros((self.max_boxes, 5))
        if len(boxes) > 0:
            np.random.shuffle(boxes)
            boxes = boxes[:self.max_boxes]
            boxes[:, 0:2] = boxes[:, 0:2] * width  # + dx
            boxes[:, 2:4] = boxes[:, 2:4] * height # + dy
            bbox_data[:len(boxes)] = boxes

        return image_data, bbox_data

