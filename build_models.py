# Importing TensorFlow & Keras libraries
import tensorflow as tf

# Import model constructors
from model_constructors.YOLOv4.yolo_model import YOLOv4


def build_model(model_name):
    """
    Function for building one of the model architectures.
    Model architectures used in this project:
    - YOLOv4;
    - SSD;
    - Faster R-CNN;
    - CenterNet (TensorHub);
    - RetinaNet (TensorHub);
    :param model_name :type string: name of the model that needs to construct;
    :return: built Keras model.
    """
    # Define model variable
    model = None
    # Assign model
    if model_name == "yolov4":
        model = YOLOv4(weight_path=None)
    # elif model_name == "ssd":
    #     model = SSD()
    # elif model_name == "rcnn":
    #     model = RCNN()
    # elif model_name == "retinanet":
    #     model = RetinaNet()
    # elif model_name == "centernet":
    #     model = CenterNet()

    return model


