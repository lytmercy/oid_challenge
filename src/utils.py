import tensorflow as tf

# Import other libraries
import pandas as pd
from pathlib import Path
import os

# Import libraries for process config YAML File
import yaml
from attrdict import AttrDict


CONFIG_PATH = "conf/"


def load_config(config_name):
    """"""
    try:
        with open(os.path.join(CONFIG_PATH, config_name), "r") as conf_file:
            config = yaml.safe_load(conf_file)
    except Exception as exc:
        raise FileNotFoundError("Error reading the config file") from exc

    return AttrDict(config)


def form_classes_code_list(class_description_path):
    """
    Forming pandas.DataFrame with classes codes and classes what they mean (classes name);
    :param class_description_path :type string: path to file with class description data;
    :return: formed pandas.DataFrame with classes codes and classes what they mean that will be used in this project;
    """

    class_description_df = pd.read_csv(class_description_path)
    class_description_df.columns = ["LabelCode", "LabelName"]

    # Prepare class code list
    classes_codes_list = class_description_df.loc[:, "LabelCode"].tolist()

    return classes_codes_list, class_description_df


def form_dataset_variables(config):
    """"""

    ds_root = Path(config.dataset.root)
    classes_path = ds_root / config.dataset.class_desc
    classes_codes_list, class_description_df = form_classes_code_list(classes_path)
    classes_names = class_description_df.loc[:, "LabelName"].tolist()

    train_bbox_path = ds_root / config.dataset.train_bbox
    gt_df = pd.read_csv(train_bbox_path, usecols=["ImageID", "LabelName",
                                                  "XMin", "XMax", "YMin", "YMax"])
    gt_df.rename(columns={"LabelName": "Label"}, inplace=True)

    train_images_path = ds_root / config.dataset.images.train
    images_paths = []
    list_ids = []
    for image_path in os.listdir(train_images_path):
        images_paths.append(train_images_path / image_path)
        list_ids.append(image_path.split(".", 1)[0])

    return classes_codes_list, class_description_df,\
        gt_df, list_ids, images_paths, classes_names


def load_prepare_image(image_path, image_size, color_channels):
    """"""

    image = tf.io.read_file(image_path)
    tensor_image_size = tf.constant(image_size)
    decoded_image = tf.image.decode_image(image, channels=color_channels)  # colour images
    # Tale actual image size
    actual_image_size = decoded_image.shape
    # Convert uint8 tensor to floats in the [0, 1] range
    decoded_image = tf.image.convert_image_dtype(decoded_image, tf.float32)
    # Resize the image into tensor_image_size
    decoded_image = tf.image.resize(decoded_image, size=tensor_image_size)

    return decoded_image, actual_image_size


