# Importing TensorFlow & Keras libraries
import tensorflow as tf

# Importing other libraries
import pandas as pd
import numpy as np

# Importing class for preprocess and getting images and labels (labels and coordinates of bbox)
from preprocessing_data import OIDDataSet

# Importing global variables from globals.py
from globals import CLASSES_NAME, CLASSES_PATH, TRAIN_DETECTION_BBOX


def train_model():
    """"""

    class_description_df = pd.read_csv(CLASSES_PATH)
    class_description_df.columns = ["LabelCode", "LabelName"]
    ground_truth_df = pd.read_csv(TRAIN_DETECTION_BBOX, usecols=["ImageID", "LabelName",
                                                                 "XMin", "XMax", "YMin", "YMax"])

    preprocess_data = OIDDataSet(ground_truth_df, class_description_df, CLASSES_NAME)
    train_dataset = preprocess_data.getting_dataset()
