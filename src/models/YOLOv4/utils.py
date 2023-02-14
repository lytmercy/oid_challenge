## Code reference:
# https://github.com/taipingeric/yolo-v4-tf.keras/blob/73dfe97c00a03ebb7fab00a5a0549b958172482a/utils.py

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
    Load weight for yolo model from file with weights;
    :param model: yolo Keras model;
    :param weights_file_path :type string: file path where is contained file with weights;
    :raise info about reading file of weights or error about unread lines.
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
    Getting prediction data from image with yolo model outputs;
    :param image: sample raw image from dataset;
    :param model_outputs: yolo inference model outputs;
    :param class_names :type list: list of object class names;
    :return: pandas.DataFrame with yolo coordinate predictions bboxes coordinate,
    class_name and score of probabilities for this class.
    """

    num_bboxes = model_outputs[-1][0]
    boxes, scores, classes = [output[0][:num_bboxes] for output in model_outputs[:-1]]

    height, width = image.shape[:2]
    df = pd.DataFrame(boxes, columns=["x1", "y1", "x2", "y2"])
    df[["x1", "x2"]] = (df[["x1", "x2"]] * width).astype("int64")
    df[["y1", "y2"]] = (df[["y1", "y2"]] * height).astype("int64")

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
    :param detections :type pandas.DataFrame: containing detections;
    :param cmap: object color map;
    :param random_color: assign random color for each object;
    :param fig_size: assign size of figure for plot;
    :param show_image: if True -- plot image with bboxes;
    :param show_text: if True -- plot text over bboxes;
    :return: processed image.
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
    Calculate the AP (Average Precision) given the recall and precision array
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

    :param rec: recall array for calculating;
    :param prec: precision array for calculating;
    :return: average precision, mean recall, mean precision.
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


def adjust_axes(r, t, fig, axes):
    """
    Adjusting axes in plot;
    :param r: renderer of plot;
    :param t: text from plot;
    :param fig: figure of plot;
    :param axes: axes of plot;
    """
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    proportion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * proportion])


def draw_plot_func(dictionary, num_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    """
    Draw plot with image with bboxes; (TP - True Positive, FP - False Positive, FN - False Negative);
    :param dictionary: dictionary with
    :param num_classes :type int: number of classes in dataset;
    :param window_title :type string: text for window title;
    :param plot_title :type string: text for plot title;
    :param x_label :type string: text for labeling x axes;
    :param output_path :type string: file path for saving formed plot;
    :param to_show :type boolean: if True then the plot will be shown;
    :param plot_color: alternate color for colouring TP, FP, FN predictions in the plot;
    :param true_p_bar: true predicted bar for colouring TP, FP and FN predictions in the plot;
    """
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
        plt.legend(loc="lower right")

        # Write number on side of bar
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color="forestgreen", va="center", fontweight="bold")
            plt.text(val, i, fp_str_val, color="crimson", va="center", fontweight="bold")
            if i == (len(sorted_values)-1):  # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(num_classes), sorted_values, color=plot_color)

        # Write number on side of bar
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)  # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va="center", fontweight="bold")
            # re-set axes to show number inside the figure
            if i == (len(sorted_values)-1):  # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y-axis
    tick_font_size = 12
    plt.yticks(range(num_classes), sorted_keys, fontsize=tick_font_size)

    # Re-scale height accordingly
    init_height = fig.get_figheight()
    # compute the matrix height in points and inches
    dpi = fig.dpi
    height_pt = num_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15  # in percentage of the figure height
    bottom_margin = 0.05  # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel("classes")
    plt.xlabel(x_label, fontsize="large")
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    # if to_show:
    plt.show()
    # close the plot
    # plt.close()


def xywh_to_x0y0x1y1(bbox):
    """
    Convert yolo bbox (center_x center_y and width height of bbox) coordinate to x1,y1,x2,y2 bbox coordinate;
    :param bbox:
    bbox[0] == (center_x) center in x-axis of bbox;
    bbox[1] == (center_y) center in y-axis of bbox;
    bbox[2] == width of bbox;
    bbox[3] == height of bbox;
    :return: bbox in x1 y1 and x2 y2 coordinates format.
    """
    return tf.concat([bbox[..., :2] - bbox[..., 2:] * 0.5,
                      bbox[..., :2] + bbox[..., 2:] * 0.5], axis=-1)


def read_txt_to_list(path):
    """
    Read txt file and convert text line to py list;
    :param path: path to file;
    :return: content from this file in py list format.
    """
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like "\n" at the end of each line
    content = [x.strip() for x in content]
    return content
