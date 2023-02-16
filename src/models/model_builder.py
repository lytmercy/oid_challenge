# Import function for load config files
from src.utils import load_config
# Import libraries for using config in elegant way
from attrdict import AttrDict

# Import model models
from src.models.YOLOv4.yolo_builder import YOLOv4


def build_model(model_name, config, model_config, classes_name):
    """
    Function for building one of the model architectures.
    Model architectures used in this project:
    - YOLOv4;
    - SSD;
    - Faster R-CNN;
    - CenterNet (TensorHub);
    - RetinaNet (TensorHub);
    :param model_name :type string: name of the model that needs to construct;
    :param config :type attrdict object: dict with default config parameters;
    :param model_config :type attrdict object: dict with model config parameters;
    :param classes_name :type list: list of classes name;
    :return: built Keras model.
    """

    # Define model variable
    model = None
    # Assign model
    match model_name:
        case "yolo": model = YOLOv4(config, model_config, classes_name, weight_path=None)
        # case "ssd": model = SSD()
        # case "rcnn": model = RCNN()
        # case "retinanet": model = RetinaNet()
        # case "centernet": model = CenterNet()

    # if model_name == "yolov4":
    #     model = YOLOv4(weight_path=None)
    # elif model_name == "ssd":
    #     model = SSD()
    # elif model_name == "rcnn":
    #     model = RCNN()
    # elif model_name == "retinanet":
    #     model = RetinaNet()
    # elif model_name == "centernet":
    #     model = CenterNet()

    return model


