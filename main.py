
# Importing function for building model_weights
from build_models import build_model
# Importing function for training model_weights
from training_model import train_model
# Importing function for take inference from trained model_weights
from inference_model import getting_inference


def main():

    # Build model_weights
    # oid_model = build_model()

    # Training model_weights
    oid_model = train_model()



if __name__ == '__main__':
    main()
