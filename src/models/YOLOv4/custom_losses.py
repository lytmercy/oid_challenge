## Code reference:
# https://github.com/taipingeric/yolo-v4-tf.keras/blob/73dfe97c00a03ebb7fab00a5a0549b958172482a/loss.py

import tensorflow as tf
import keras.backend as K

# Import other libraries
import numpy as np
import sys

# Import utils function for loss function
from src.models.YOLOv4.utils import xywh_to_x0y0x1y1


def bbox_iou(boxes1, boxes2):
    """
    Calculate Intersection Over Union loss for yolo (xy & wh) coordinates;
    :param boxes1: first bbox for calculating (default prediction);
    :param boxes2: second bbox for calculating default ground truth);
    :return: calculated IoU.
    """
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # w * h
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # (x,y,w,h) -> (x0,y0,x1,y1)
    boxes1 = xywh_to_x0y0x1y1(boxes1)
    boxes2 = xywh_to_x0y0x1y1(boxes2)

    # coordinates of intersection
    top_left = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection_xy = tf.maximum(bottom_right - top_left, 0.0)

    intersection_area = intersection_xy[..., 0] * intersection_xy[..., 1]
    union_area = boxes1_area + boxes2_area - intersection_area

    iou = 1.0 * intersection_area / (union_area + K.epsilon())

    return iou, union_area


def bbox_giou(boxes1, boxes2):
    """
    Calculate generalized Intersection Over Union loss for yolo (xy & wh) coordinates;
    :param boxes1: first bbox for calculating (default prediction);
    :param boxes2: second bbox for calculating (default ground truth);
    :return: calculated GIoU.
    """
    # boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # w * h
    # boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    #
    # # (x,y,w,h) -> (x0,y0,x1,y1)
    # boxes1 = xywh_to_x0y0x1y1(boxes1)
    # boxes2 = xywh_to_x0y0x1y1(boxes2)
    #
    # # coordinates of intersection
    # top_left = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    # bottom_right = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    # intersection_xy = tf.maximum(bottom_right - top_left, 0.0)
    # intersection_area = intersection_xy[..., 0] * intersection_xy[..., 1]
    #
    # union_area = boxes1_area + boxes2_area - intersection_area
    #
    # # Define iou
    # iou = 1.0 * intersection_area / (union_area + K.epsilon())

    iou, union_area = bbox_iou(boxes1, boxes2)

    enclose_top_left = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_bottom_right = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])

    enclose_xy = enclose_bottom_right - enclose_top_left
    enclose_area = enclose_xy[..., 0] * enclose_xy[..., 1]
    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)

    return giou


def decode(conv_output, anchors, stride, num_classes):
    """
    Decoding raw convolution prediction data to xy wh coordinates, confidence and probabilities;
    :param conv_output: keras convolution layer outputs;
    :param anchors :type list: anchors for yolo model;
    :param stride: stride for making prediction xy coordinate of bbox;
    :param num_classes :type int: number of classes in dataset;
    :return: concatenated, in one tensor, predictions of xy wh coordinate, confidence and probabilities.
    """
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    anchor_per_scale = len(anchors)
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes))
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def loss_layer(conv, pred, label, bboxes, stride, num_classes, iou_loss_thresh):
    """
    Custom loss layer for YOLO loss function;
    :param conv: convolution Keras layer;
    :param pred: predictions bbox from model;
    :param label: labeled bbox from model;
    :param bboxes: bboxes from model;
    :param stride: stride for changing output_size from conv layer;
    :param num_classes :type int: number of classes in dataset;
    :param iou_loss_thresh: minimum overlap that counts as a valid detection;
    :returns: ciou, confidence and probabilities losses.
    """
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = stride * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + num_classes))

    conv_raw_prob = conv[:, :, :, :, 5:]
    conv_raw_conf = conv[:, :, :, :, 4:5]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    # Coordinate loss
    ciou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)  # (8, 13, 13, 3, 1)
    input_size = tf.cast(input_size, tf.float32)

    # loss weight of the gt bbox: 2 - (gt area / img area)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)  # iou loss for respond bbox

    # Classification loss for respond bbox
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    expand_pred_xywh = pred_xywh[:, :, :, :, np.newaxis, :]  # (?, grid_h, grid_w, 3, 1, 4)
    expand_bboxes = bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :]  # (?, 1, 1, 1, 70, 4)
    iou = bbox_iou(expand_pred_xywh, expand_bboxes)  # IoU between all pred bbox and all gt (?, grid_h, grid_w, 3, 70)
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)  # max iou: (?, grid_h, grid_w, 3, 1)

    # ignore the bbox which is not respond bbox and max iou < threshold
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < iou_loss_thresh, tf.float32)

    # Confidence loss
    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
        respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        +
        respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    ciou_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return ciou_loss, conf_loss, prob_loss


def yolo_loss(args, num_classes, iou_loss_thresh, anchors):
    """
    Custom YOLO loss function for YOLO model;
    :param args: arguments with convolution, label for small, medium and large bboxes, and true bboxes;
    :param num_classes :type int: number of classes in dataset;
    :param iou_loss_thresh :type float: minimum overlap that counts as a valid detection;
    :param anchors :type list: anchors for yolo model;
    :returns: sum of ciou, confidence and probabilities for yolo loss;
    """
    conv_sbbox = args[0]
    conv_mbbox = args[1]
    conv_lbbox = args[2]

    label_sbbox = args[3]
    label_mbbox = args[4]
    label_lbbox = args[5]

    true_bboxes = args[6]

    pred_sbbox = decode(conv_sbbox, anchors[0], 8, num_classes)
    pred_mbbox = decode(conv_mbbox, anchors[0], 8, num_classes)
    pred_lbbox = decode(conv_lbbox, anchors[0], 8, num_classes)

    sbbox_ciou_loss, sbbox_conf_loss, sbbox_prob_loss = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_bboxes, 8, num_classes, iou_loss_thresh)
    mbbox_ciou_loss, mbbox_conf_loss, mbbox_prob_loss = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_bboxes, 16, num_classes, iou_loss_thresh)
    lbbox_ciou_loss, lbbox_conf_loss, lbbox_prob_loss = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_bboxes, 32, num_classes, iou_loss_thresh)

    ciou_loss = (sbbox_ciou_loss + mbbox_ciou_loss + lbbox_ciou_loss) * 3.54
    conf_loss = (sbbox_conf_loss + mbbox_conf_loss + lbbox_conf_loss) * 64.3
    prob_loss = (sbbox_prob_loss + mbbox_prob_loss + lbbox_prob_loss) * 1

    return ciou_loss + conf_loss + prob_loss
