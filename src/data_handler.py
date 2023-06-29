# Importing TensorFlow & Keras libraries
import tensorflow as tf
from keras.utils import image_utils, dataset_utils, Sequence

# Importing other libraries
import numpy as np
import pandas as pd
from pathlib import Path
from attrdict import AttrDict
import os

# Import utils functions
from src.utils import load_prepare_image


def preprocess_true_boxes(true_boxes: np.ndarray,
                          input_shape: tuple,
                          anchors: list,
                          num_classes: int) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Preprocess true boxes to training input format;
    :param true_boxes: shape=(batch_size, max_boxes_per_image, 5);
    Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape;
    :param input_shape: height & width of image;
    :param anchors: anchors for yolo model (shape=(N, 2), (9, width height));
    :param num_classes: number of classes from dataset;
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
    grid_sizes = [input_shape//{0: 8, 1: 16, 2: 32}[stage] for stage in range(num_stages)]
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
        if num_boxes == 0:
            continue
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
                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 4] = 1  # confidence

                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 5 + class_idx] = 1  # one-hot encoding

    return y_true, y_true_boxes_xywh


class OIDDataset(Sequence):
    """"""
    def __init__(self,
                 subset: str,
                 list_ids: list,
                 ground_truth_df: pd.DataFrame,
                 classes_codes_list: list,
                 config: AttrDict,
                 model_config: AttrDict,
                 mode: str = "fit",
                 color_channels: int = 3,
                 random_seed: int = 17,
                 shuffle: bool = True):
        """
        Init method for remaining all information about a dataset;
        :param ground_truth_df: pandas DataFrame that contains all names & boundary boxes of the image;
        :param classes_codes_list: that contains a list of needed classes for this project;

        :param color_channels: int number of colour encoding channels in an image;
        :param random_seed: int number of random state for getting same result from dataset everytime;
        :param shuffle: boolean variable that says shuffle this dataset or not;
        """

        self.config = config
        self.model_config = model_config
        self.list_ids = list_ids
        self.gt_df = ground_truth_df
        self.anchors = np.array(model_config.anchors).reshape((9, 2))
        self.classes_codes_list = classes_codes_list
        self.num_classes = len(classes_codes_list)
        self.color_channels = color_channels
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.mode = mode
        self.num_gpu = config.num_gpu
        self.images_root = Path(self.config.dataset.root) / subset

        self.batch_size = self.config.preprocess.batch_size
        self.image_size = self.model_config.image_size
        self.max_boxes = self.model_config.max_boxes

        self.images_paths = []
        for image_id in os.listdir(self.images_root):
            self.images_paths.append(self.images_root / image_id)
        self.images_paths = np.array(self.images_paths)

        # init indexes for store all indexes for loading to batch
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def on_epoch_end(self):
        """"""
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle is True:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.indexes)

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

    def __data_generation(self, list_ids_batch):
        """"""
        X = np.empty((len(list_ids_batch), *self.image_size + (self.color_channels,)), dtype=np.float32)

        y_bbox = np.empty((len(list_ids_batch), self.max_boxes, 5), dtype=np.float32)  # x1y1x2y2

        for i, image_id in enumerate(list_ids_batch):
            image_data, bbox_data = self.get_data(image_id, self.images_paths[i])
            X[i] = image_data
            y_bbox[i] = bbox_data

        y_tensor, y_true_boxes_xywh = preprocess_true_boxes(y_bbox, self.image_size, self.anchors, self.num_classes)

        return X, y_tensor, y_true_boxes_xywh

    def get_data(self, image_id, image_path):
        """"""
        # Take image and actual image size from image path
        image_data, actual_image_size = load_prepare_image(str(image_path), self.image_size, self.color_channels)

        # Make scale number for scaling bbox for changed image size
        height, width = self.image_size
        image_height, image_width = actual_image_size[0], actual_image_size[1]
        scale_width, scale_height = width / image_width, height / image_height

        bboxes_df = self.gt_df.loc[self.gt_df["ImageID"] == image_id]
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
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_height  # + dy
            bbox_data[:len(boxes)] = boxes

        return image_data, bbox_data
