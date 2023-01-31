import tensorflow as tf
from keras.utils import Sequence

# Import other libraries
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import operator
import os


def load_weights(model, weights_file_path):
    """

    :param model:
    :param weights_file_path:
    :raise info about reading file of weights.
    """

    conv_layer_size = 110
    conv_output_idxs = [93, 101, 109]
    with open(weights_file_path, "rb") as file:
        major, minor, revision, seen, _ = np.fromfile(file, dtype=np.int32, count=5)

        bn_idx = 0
        for conv_idx in range(conv_layer_size):
            conv_layer_name = f"conv2d_{conv_idx}" if conv_idx > 0 else "conv2d"
            bn_layer_name = f"batch_normalization_{bn_idx}" if bn_idx > 0 else "batch_normalization"

            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            kernel_size = conv_layer.kernel_size[0]
            input_dims = conv_layer.input_shape[-1]

            if conv_idx not in conv_output_idxs:
                # darknet bn layer weights: [beta, gamma, mean, variance]
                bn_weights = np.fromfile(file, dtype=np.float32, count=4 * filters)
                # tf bn layer weights: [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                bn_idx += 1
            else:
                conv_bias = np.fromfile(file, dtype=np.float32, count=filters)

            # darknet shape: (out_dim, input_dims, height, width)
            # tf shape: (height, width, input_dims, out_dim)
            conv_shape = (filters, input_dims, kernel_size, kernel_size)
            conv_weights = np.fromfile(file, dtype=np.float32, count=np.product(conv_shape))
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if conv_idx not in conv_output_idxs:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        if len(file.read()) == 0:
            print("all weights read")
        else:
            print(f"failed to read all weights, # of unread weights: {len(file.read())}")


def get_detection_data(image, model_outputs, class_names):
    """

    :param image:
    :param model_outputs:
    :param class_names:
    :return:
    """

    num_bboxes = model_outputs[-1][0]
    boxes, scores, classes = [output[0][:num_bboxes] for output in model_outputs[:-1]]

    h, w = image.shape[:2]
    df = pd.DataFrame(boxes, columns=["x1", "y1", "x2", "y2"])
    df[["x1", "x2"]] = (df[["x1", "x2"]] * w).astype("int64")
    df[["y1", "y2"]] = (df[["y1", "y2"]] * h).astype("int64")

    df["class_name"] = np.array(class_names)[classes.astype("int64")]
    df["score"] = scores

    df["w"] = df["x2"] - df ["x1"]
    df["h"] = df["y2"] - df ["y1"]

    print(f"# of bboxes: {num_bboxes}")
    return df


def draw_bbox(image, detections, cmap, random_color=True, fig_size=(10, 10), show_image=True, show_text=True):
    """
    Draw bounding boxes on the image;
    :param image: BGR image;
    :param detections:
    :param cmap:
    :param random_color:
    :param fig_size:
    :param show_image:
    :param show_text:
    :return:
    """

    image = np.array(image)
    scale = max(image.shape[0:2]) / 416
    line_width = int(2 * scale)

    for _, row in detections.iterrows():
        x1, y1, x2, y2, clss, score, w, h = row.values
        color = list(np.random.random(size=3) * 255) if random_color else cmap[clss]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)
        if show_text:
            text = f"{clss} {score:.2f}"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = max(0.3 * scale, 0.3)
            thickness = max(int(1 * scale), 1)
            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
            cv2.rectangle(image, (x1 - line_width//2, y1 - text_height), (x1 + text_width, y1), color, cv2.FILLED)
            cv2.putText(image, text, (x1, y1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if show_image:
        plt.figure(figsize=fig_size)
        plt.imshow(image)
        plt.show()

    return image


def voc_ap(rec, prec):
    """
    Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.

    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));

    :param rec:
    :param prec:
    :return:
    """

    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]

    """This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """

    # matlab indexes start in 1 but python in 0, so I have to do:
    #       range(start=(len(mpre) - 2), end=0, stepa=-1)
    # also the python function range excludes the end, resulting in:
    #       range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    """This part creates a list of indexes where the recall changes
            matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i-1]) * mpre[i])

    return ap, mrec, mpre


def draw_plot_func(dictionary, num_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    print(sorted_dic_by_value)
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)

    if true_p_bar != "":
        """ Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - pink -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])

        plt.barh(range(num_classes), fp_sorted, align="center", color="crimson", label="False Positive")
        plt.barh(range(num_classes), fp_sorted, align="center", color="forestgreen", label="True Positive", left=fp_sorted)

        # add legend




