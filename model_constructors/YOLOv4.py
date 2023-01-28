import tensorflow as tf
from keras.initializers import initializers_v2 as initializers
from keras.layers import Input, Dense, ZeroPadding2D, Conv2D
from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, LeakyReLU, Add, Concatenate
from keras.models import Model

# Importing other libraries
import numpy as np



def mish(x):
    """
    Activation function for YOLO Convolution Layer;
    :param x: Keras Conv Layer;
    :return: mish activation function.
    """
    return x * tf.math.tanh(tf.math.softplus(x))


def yolo_convolution(x, filters, kernel_size, downsampling=False, activation='leaky', batch_norm=True):
    """
    Function for forming Convolution Layer for YOLO model;
    :param x: Keras previous layer;
    :param filters: int, number of filters for convolution layer;
    :param kernel_size: int, kernel size of filters for convolution layer;
    :param downsampling: boolean, variable for deciding to make downsampling or not before the convolution layer;
    :param activation: string, name of activation function for convolution layer;
    :param batch_norm: boolean, variable for deciding to make batch-normalization or not after convolution layer;
    :return: new Keras Convolution Layer with previous layers.
    """

    if downsampling:
        x = ZeroPadding2D(padding=((1, 0), (1, 0)))(x)  # top & left padding
        padding = "valid"
        strides = 2
    else:
        padding = "same"
        strides = 1

    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=not batch_norm,
               # kernel_regularizer=regularizers.l2(0.0005),
               kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
               # bias_initializer=initializers.Zeros()
               )(x)

    if batch_norm:
        x = BatchNormalization()(x)
    if activation == "mish":
        x = mish(x)
    elif activation == "leaky":
        x = LeakyReLU(alpha=0.1)(x)

    return x


def yolo_residual_block(x, filters1, filters2, activation="leaky"):
    """
    Residual block for using in YOLO model architecture;
    :param x: input tensor;
    :param filters1: int, num of filter for 1x1 conv;
    :param filters2: int, num of filter for 3x3 conv;
    :param activation: string, activation function for model, default: leaky relu;
    :return: Keras "Add" layer with x (previous layer) and y (new layer).
    """
    y = yolo_convolution(x, filters1, kernel_size=1, activation=activation)
    y = yolo_convolution(y, filters2, kernel_size=3, activation=activation)
    return Add()([x, y])


def csp_block(x, residual_out, repeat, residual_bottleneck=False):
    """
    Cross Stage Partial Network (CSPNet);
    transition_bottleneck_dims: 1x1 bottleneck;
    output_dims: 3x3;
    :param x: Keras previous layer;
    :param residual_out: int, size of residual block;
    :param repeat: int, number of repeat residual block;
    :param residual_bottleneck: boolean, for deciding make bottleneck in residual block or not;
    :return: new concatenate layer with previous x layers new residual block layer.
    """
    route = x
    route = yolo_convolution(route, residual_out, 1, activation="mish")
    x = yolo_convolution(x, residual_out, 1, activation="mish")
    for i in range(repeat):
        x = yolo_residual_block(x,
                                residual_out // 2 if residual_bottleneck else residual_out,
                                residual_out,
                                activation="mish")

    x = yolo_convolution(x, residual_out, 1, activation="mish")

    x = Concatenate()([x, route])
    return x


def csp_darknet53(input):
    x = yolo_convolution(input, 32, 3)
    x = yolo_convolution(x, 64, 3, downsampling=True)

    x = csp_block(x, residual_out=64, repeat=1, residual_bottleneck=True)
    x = yolo_convolution(x, 64, 1, activation="mish")
    x = yolo_convolution(x, 128, 3, activation="mish", downsampling=True)

    x = csp_block(x, residual_out=64, repeat=2)
    x = yolo_convolution(x, 128, 1, activation="mish")
    x = yolo_convolution(x, 256, 3, activation="mish", downsampling=True)

    x = csp_block(x, residual_out=128, repeat=8)
    x = yolo_convolution(x, 256, 1, activation="mish")
    route0 = x
    x = yolo_convolution(x, 512, 3, activation="mish", downsampling=True)

    x = csp_block(x, residual_out=256, repeat=8)
    x = yolo_convolution(x, 512, 1, activation="mish")
    route1 = x
    x = yolo_convolution(x, 1024, 3, activation="mish", downsampling=True)

    x = csp_block(x, residual_out=512, repeat=4)

    x = yolo_convolution(x, 1024, 1, activation="mish")

    x = yolo_convolution(x, 512, 1)
    x = yolo_convolution(x, 1024, 3)
    x = yolo_convolution(x, 512, 1)

    x = Concatenate()([MaxPooling2D(pool_size=13, strides=1, padding="same")(x),
                       MaxPooling2D(pool_size=9, strides=1, padding="same")(x),
                       MaxPooling2D(pool_size=9, strides=1, padding="same")(x),
                       x])

    x = yolo_convolution(x, 512, 1)
    x = yolo_convolution(x, 1024, 3)
    route2 = yolo_convolution(x, 512, 1)

    return Model(input, [route0, route1, route2])


def get_boxes(pred, anchors, num_classes, grid_size, strides, xy_scale):
    """

    :param pred:
    :param anchors:
    :param num_classes:
    :param grid_size:
    :param strides:
    :param xy_scale:
    :return:
    """
    pred = tf.reshape(pred,
                      (tf.shape(pred)[0],  # batch_size
                       grid_size,
                       grid_size,
                       3,
                       5 + num_classes))
    box_xy, box_wh, obj_prob, class_prob = tf.split(
        pred, (2, 2, 1, num_classes), axis=-1
    )  # (batch, 52, 52, 3, 2) (batch, 52, 52, 3, 2) (batch, 52, 52, 3, 1) (batch, 52, 52, 3, 80)

    box_xy = tf.sigmoid(box_xy)
    obj_prob = tf.sigmoid(obj_prob)
    class_prob = tf.sigmoid(class_prob)
    pred_box_xywh = tf.concat((box_xy, box_wh), axis=-1)

    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    grid = tf.cast(grid, dtype=tf.float32)

    box_xy = ((box_xy * xy_scale) - 0.5 * (xy_scale - 1) + grid) * strides

    box_wh = tf.exp(box_wh) * anchors
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    # pred_box_x1y1x2y2: absolute xy value
    pred_box_x1y1x2y2 = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return pred_box_x1y1x2y2, obj_prob, class_prob, pred_box_xywh


def non_max_suppression(model_outputs, image_size, num_classes, iou_threshold, score_threshold=0.3):
    """
    Apply Non-Maximum suppression;
    ref: https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
    :param model_outputs:
    :param image_size:
    :param num_classes:
    :param iou_threshold:
    :param score_threshold:
    :return:
    """
    bs = tf.shape(model_outputs[0])[0]
    boxes = tf.zeros((bs, 0, 4))
    confidence = tf.zeros((bs, 0, 1))
    class_probabilities = tf.zeros((bs, 0, num_classes))

    for output_idx in range(0, len(model_outputs), 4):
        output_xy = model_outputs[output_idx]
        output_conf = model_outputs[output_idx + 1]
        output_classes = model_outputs[output_idx + 2]
        boxes = tf.concat([boxes, tf.reshape(output_xy, (bs, -1, 4))], axis=-1)
        confidence = tf.concat([confidence, tf.reshape(output_conf, (bs, -1, 1))], axis=1)
        class_probabilities = tf.concat([class_probabilities, tf.reshape(output_classes, (bs, -1, num_classes))], axis=1)

    scores = confidence * class_probabilities
    boxes = tf.expand_dims(boxes, axis=-2)
    boxes = boxes / image_size[0]  # box normalization: relative image size
    print(f"non-max iou: {iou_threshold} score: {score_threshold}")
    (non_max_sed_boxes,     # [bs, max_detections, 4]
     non_max_sed_scores,    # [bs, max_detections]
     non_max_sed_classes,   # [bs, max_detections]
     valid_detections       # [batch_size]
     ) = tf.image.combined_non_max_suppression(
        boxes=boxes,  # y1x1, y2x2 [0~1]
        scores=scores,
        max_output_size_per_class=100,
        max_total_size=100,  # max_boxes: Maximum non_max_sed_boxes in a single img.
        iou_threshold=iou_threshold,  # iou_threshold: Minimum overlap that counts as a valid detection.
        score_threshold=score_threshold,  # Minimum confidence that counts as a valid detection.
    )

    return non_max_sed_boxes, non_max_sed_scores, non_max_sed_classes, valid_detections



class YOLOv4:
    """Class of YOLOv4 model_weights for build and entire forming this model_weights."""
    def __init__(self, image_size, num_classes, batch_size, iou_threshold=0.413, score_threshold=0.3):
        """"""
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def neck(self, x):
        """Method for build model_weights neck."""
        # Define model backbone
        backbone_model = csp_darknet53(x)
        # Define output of model backbone
        route0, route1, route2 = backbone_model.output

        route_input = route2
        x = yolo_convolution(route2, 256, 1)
        x = UpSampling2D()(x)
        route1 = yolo_convolution(route1, 256, 1)
        x = Concatenate()([route1, x])

        x = yolo_convolution(x, 256, 1)
        x = yolo_convolution(x, 512, 3)
        x = yolo_convolution(x, 256, 1)
        x = yolo_convolution(x, 512, 3)
        x = yolo_convolution(x, 256, 1)

        route1 = x
        x = yolo_convolution(x, 128, 1)
        x = UpSampling2D()(x)
        route0 = yolo_convolution(route0, 128, 1)
        x = Concatenate()([route0, x])

        x = yolo_convolution(x, 128, 1)
        x = yolo_convolution(x, 256, 3)
        x = yolo_convolution(x, 128, 1)
        x = yolo_convolution(x, 256, 3)
        x = yolo_convolution(x, 128, 1)

        route0 = x
        x = yolo_convolution(x, 256, 3)
        conv_sbbox = yolo_convolution(x, 3 * (self.num_classes + 5), 1, activation=None, batch_norm=False)

        x = yolo_convolution(route1, 256, 3, downsampling=True)
        x = Concatenate()([x, route1])

        x = yolo_convolution(x, 256, 1)
        x = yolo_convolution(x, 512, 3)
        x = yolo_convolution(x, 256, 1)
        x = yolo_convolution(x, 512, 3)
        x = yolo_convolution(x, 256, 1)

        route1 = x
        x = yolo_convolution(x, 512, 3)
        conv_mbbox = yolo_convolution(x, 3 * (self.num_classes + 5), 1, activation=None, batch_norm=False)

        x = yolo_convolution(route1, 512, 3, downsampling=True)
        x = Concatenate()([x, route_input])

        x = yolo_convolution(x, 512, 1)
        x = yolo_convolution(x, 1024, 3)
        x = yolo_convolution(x, 512, 1)
        x = yolo_convolution(x, 1024, 3)
        x = yolo_convolution(x, 512, 1)

        x = yolo_convolution(x, 1024, 3)
        conv_lbbox = yolo_convolution(x, 3 * (self.num_classes + 5), 1, activation=None, batch_norm=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]

    def heads(self, yolo_neck_outputs, anchors, xy_scale):
        """
        Method for build model_weights heads
        :param yolo_neck_outputs:
        :param anchors:
        :param xy_scale:
        :return:
        """
        bbox0, object_probability0, class_probabilities0, pred_box0 = get_boxes(yolo_neck_outputs[0],
                                                                                anchors=anchors[0, :, :],
                                                                                num_classes=self.num_classes,
                                                                                grid_size=52, strides=8,
                                                                                xy_scale=xy_scale[0])
        bbox1, object_probability1, class_probabilities1, pred_box1 = get_boxes(yolo_neck_outputs[1],
                                                                                anchors=anchors[1, :, :],
                                                                                num_classes=self.num_classes,
                                                                                grid_size=26, strides=16,
                                                                                xy_scale=xy_scale[1])
        bbox2, object_probability2, class_probabilities2, pred_box2 = get_boxes(yolo_neck_outputs[2],
                                                                                anchors=anchors[2, :, :],
                                                                                num_classes=self.num_classes,
                                                                                grid_size=13, strides=32,
                                                                                xy_scale=xy_scale[2])

        x = [bbox0, object_probability0, class_probabilities0, pred_box0,
             bbox1, object_probability1, class_probabilities1, pred_box1,
             bbox2, object_probability2, class_probabilities2, pred_box2]

        return x

    def build_yolo(self):
        """Method for entire forming of YOLO model_weights with all components (backbone, neck and heads)."""


