
# Importing function for building model_weights
from build_models import build_model
# Importing function for training model_weights
from training_model import train_model
# Importing function for take inference from trained model_weights
from inference_model import getting_inference


def main():

    # Build model
    yolo_model = build_model("yolov4")

    # Training model
    yolo_model = train_model(yolo_model)

    # Evaluate model

    # Test model



if __name__ == '__main__':
    main()
