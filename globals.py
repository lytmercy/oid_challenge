
"""This file contain all global variables for this project"""
# Initialize globals
global BATCH_SIZE
global IMAGE_SIZE
global EPOCHS
global CLASSES_NAME

global CLASSES_PATH
global TRAIN_DIR
global TRAIN_DETECTION_BBOX
global TRAIN_DETECTION_IMAGE_LABELS
global TEST_DIR
global TEST_DETECTION_BBOX
global TEST_DETECTION_IMAGE_LABELS

global WEIGHT_CHECKPOINT_PATH
global MODEL_WEIGHT_SAVING_PATH

# Set globals
BATCH_SIZE = 22
IMAGE_SIZE = (224, 224)
EPOCHS = 5
CLASSES_NAME = ["Bird", "Bear", "Cat"]

CLASSES_PATH = "dataset\\challenge-2019-classes-description-500.csv"
TRAIN_DIR = "dataset\\train\\"
TRAIN_DETECTION_BBOX = "dataset\\challenge-2019-train-detection-bbox.csv"
TRAIN_DETECTION_IMAGE_LABELS = "dataset\\challenge-2019-train-detection-human-imagelabels.csv"
TEST_DIR = "dataset\\test\\"
TEST_DETECTION_BBOX = "dataset\\challenge-2019-validation-detection-bbox.csv"
TEST_DETECTION_IMAGE_LABELS = "dataset\\challenge-2019-validation-detection-human-imagelabels.csv"

WEIGHT_CHECKPOINT_PATH = "model\\oid_checkpoint_weight\\oid_weight.ckpt"
MODEL_WEIGHT_SAVING_PATH = "model\\oid_model_weight\\oid_model"
