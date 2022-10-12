# Importing TensorFlow & Keras libraries
import tensorflow as tf
from keras.layers import Input, Dense
from keras import Model
from keras.applications import MobileNetV2

# Importing other libraries
import numpy as np


def build_model(image_size, num_classes):
    """"""

    # === Build YOLO Architecture model ===

    # Initialize Input Layer
    inputs = Input(shape=image_size + (3,))


