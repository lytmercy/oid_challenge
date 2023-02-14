# Importing TensorFlow & Keras libraries
import tensorflow as tf
from keras.utils import image_utils, dataset_utils, Sequence

# Importing other libraries
import numpy as np
import pandas as pd
import os

# Importing global variables
from globals import BATCH_SIZE, IMAGE_SIZE, MAX_BBOXES
from src.models.configs import yolo_config


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
    # Tale actual image size
    actual_image_size = decoded_image.shape
    # Convert uint8 tensor to floats in the [0, 1] range
    decoded_image = tf.image.convert_image_dtype(decoded_image, tf.float32)
    # Resize the image into image_size
    decoded_image = tf.image.resize(decoded_image, size=tensor_image_size)

    return decoded_image, actual_image_size


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
    true_boxes_xy = (true_boxes_abs[..., 0:2] + true_boxes_abs[..., 2:4]) // 2  # (50, 2)
    true_boxes_wh = true_boxes_abs[..., 2:4] - true_boxes_abs[..., 0:2]  # (50, 2)

    # Normalize x,y,w,h relative to image size -> (0~1)
    true_boxes[..., 0:2] = true_boxes_xy/input_shape[::-1]  # xy
    true_boxes[..., 2:4] = true_boxes_wh/input_shape[::-1]  # wh

    batch_size = true_boxes.shape[0]
    grid_sizes = [input_shape//{0:8, 1:16, 2:32}[stage] for stage in range(num_stages)]
    # [(?, 52, 52, 3, 5 + num_classes) (?, 26, 26, 3, 5 + num_classes) (?, 13, 13, 3, 5 + num_classes)]
    y_true = [np.zeros((batch_size,
                        grid_sizes[stage][0],
                        grid_sizes[stage][1],
                        bbox_per_grid,
                        5+num_classes), dtype="float32")
              for stage in range(num_stages)]


    y_true_boxes_xywh = np.concatenate((true_boxes_xy, true_boxes_wh), axis=-1)
    # Expand dim to apply broadcasting
    anchors = np.expand_dims(anchors, 0)  # (1, 9, 2)
    anchor_maxes = anchors / 2.  # (1, 9, 2)
    anchor_mins = -anchor_maxes  # (1, 9, 2)
    valid_mask = true_boxes_wh[..., 0] > 0  # (1, 50)

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
                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 2:4] = true_boxes_wh[batch_idx, box_idx, :]  # abs wh
                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 4] = 1 # confidence

                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 5 + class_idx] = 1  # one-hot encoding

    return y_true, y_true_boxes_xywh


class OIDDataset(Sequence):
    """"""
    def __init__(self, subset, list_ids, ground_truth_df, classes_codes_list, class_description_df, mode="fit",
                 model_type="yolo", label_mode="binary", color_channels=3, random_seed=17, shuffle=True,
                 interpolation="bilinear"):
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
        self.color_channels = color_channels
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.mode = mode
        self.num_gpu = 1
        self.anchors = np.array(yolo_config["anchors"]).reshape((9, 2))

        if model_type == "yolo":
            self.batch_size = yolo_config["batch_size"]
            self.image_size = yolo_config["image_size"][:2]
            self.max_boxes = yolo_config["max_boxes"]
        else:
            self.batch_size = BATCH_SIZE
            self.image_size = IMAGE_SIZE
            self.max_boxes = MAX_BBOXES

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
        X, y_tensor, y_bbox = self.__data_generation(list_ids_batch)

        # Check string attribute mode for "fit" mode
        if self.mode == "fit":
            # Return X (images) and y (ground truth -> boundary box)
            return [X, *y_tensor, y_bbox], np.zeros(len(list_ids_batch))
        # Check string attribute mode for "predict" model
        elif self.mode == "predict":
            # Return only one X (images)
            return X, np.zeros(len(list_ids_batch))
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

        # Take image and actual image size from image path
        image_data, actual_image_size = load_prepare_image(image_path, self.image_size)

        # Make scale number for scaling bbox for changed image size
        height, width = self.image_size
        image_height, image_width = actual_image_size[0], actual_image_size[1]
        scale_width, scale_height = width / image_width, height / image_height

        bboxes_df = self.ground_truth_df.loc[self.ground_truth_df["ImageID"] == image_id]
        bboxes_df = bboxes_df.loc[bboxes_df["Label"].isin(self.classes_codes_list),
                                  ["XMin", "YMin", "XMax", "YMax", "Label"]]

        # Correct bboxes label
        replace_dict = {class_code: class_number for class_number, class_code in enumerate(self.classes_codes_list)}
        bboxes_df["Label"] = bboxes_df.Label.replace(replace_dict)

        # Define bboxes list
        boxes = np.array(bboxes_df.values.tolist())

        # Denormalize coordinate
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * image_width
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * image_height

        # Correct bboxes coordinates
        bbox_data = np.zeros((self.max_boxes, 5))
        if len(boxes) > 0:
            np.random.shuffle(boxes)
            boxes = boxes[:self.max_boxes]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_width  # + dx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_height # + dy
            bbox_data[:len(boxes)] = boxes

        return image_data, bbox_data

