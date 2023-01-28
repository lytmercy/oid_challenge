# Importing TensorFlow & Keras libraries
import tensorflow as tf

# Import model constructors
from model_constructors.YOLOv4 import YOLOv4


def build_model(image_size, num_classes, model_name):
    """
    Function for building one of the model_weights architectures.
    Model architectures used in this project:
    - YOLOv4;
    - SSD;
    - Faster R-CNN;
    - CenterNet (TensorHub);
    - RetinaNet (TensorHub);
    :param image_size:
    :param num_classes:
    :param model_name:
    :return:
    """

    # === Build YOLO Architecture model_weights ===
