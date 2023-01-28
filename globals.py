
"""This file contain all global variables for this project"""
# Initialize globals
global BATCH_SIZE
global IMAGE_SIZE
global EPOCHS
global CLASSES_NAME
global MAX_BBOXES

global CLASSES_PATH
global TRAIN_DIR
global TRAIN_DETECTION_BBOX
global TRAIN_DETECTION_IMAGE_LABELS
global VALIDATION_DIR
global VALIDATION_DETECTION_BBOX
global VALIDATION_DETECTION_IMAGE_LABELS
global TEST_DIR
global TEST_DETECTION_BBOX
global TEST_DETECTION_IMAGE_LABELS

global WEIGHT_CHECKPOINT_PATH
global MODEL_WEIGHT_SAVING_PATH

global ALLOW_IMAGE_FORMATS

# Set globals
BATCH_SIZE = 22
IMAGE_SIZE = (224, 224)
EPOCHS = 5
CLASSES_NAME = ["Bear", "Bird", "Cat"]
MAX_BBOXES = 10

CLASSES_PATH = "dataset\\class-descriptions.csv"
TRAIN_DIR = "dataset\\train"
TRAIN_DETECTION_BBOX = "dataset\\train-annotations-bbox.csv"
TRAIN_DETECTION_IMAGE_LABELS = "dataset\\train-images.csv"
VALIDATION_DIR = "dataset\\validation"
VALIDATION_DETECTION_BBOX = "dataset\\validation-annotations-bbox.csv"
VALIDATION_DETECTION_IMAGE_LABELS = "dataset\\validation-images.csv"
TEST_DIR = "dataset\\test"
TEST_DETECTION_BBOX = "dataset\\test-annotations-bbox.csv"
TEST_DETECTION_IMAGE_LABELS = "dataset\\test-images.csv"

WEIGHT_CHECKPOINT_PATH = "model_weights\\oid_checkpoint_weight\\oid_weight.ckpt"
MODEL_WEIGHT_SAVING_PATH = "model_weights\\oid_model_weight\\oid_model"

ALLOW_IMAGE_FORMATS = (".jpg", ".jpeg", "png")
