
# Importing function for building model
from build_model import build_model
# Importing function for training model
from training_model import train_model
# Importing function for take inference from trained model
from inference_model import getting_inference


def main():

    # Build model
    # oid_model = build_model()

    # Training model
    oid_model = train_model()



if __name__ == '__main__':
    main()
