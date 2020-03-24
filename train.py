import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from src.m4_generator import M4Generator
from src.m4_data_loader import M4DataLoader
from src.m4_evaluation_loss_functions import *
from src.training_loss_functions import *
from src.visualization import *
from src.evaluation import *
from src.data_augmentations import *
from src.m4_model import M4Model

keras.backend.set_floatx('float64')


EPOCHS = 5
BATCH_SIZE = 128
LOOKBACK = 48
HORIZON = 48
HIDDEN_LAYER_SIZE = 6
HIDDEN_LAYERS = 6
FEATURES_NUMBER = 2
CLIP_VALUE = 2
LEARNING_RATE = 0.01
DROPOUT_RATIO = 0.2

OUTPUT_SIZE = HORIZON*2
LOSS = kl_divergance
PI_PARAMS = {'max_coff': 0.25, 'min_coff': 0.15, 'step': 0.1}

stdAugmentation = StdAugmentation(PI_PARAMS)
diffAugmentation = DiffAugmentation()
x_augmentations = [stdAugmentation]
y_augmentations = [stdAugmentation]

TRAIN_PATH = "Dataset/splitted/Hourly-train.csv"
TEST_PATH = "Dataset/splitted/Hourly-test.csv"
TRAIN_HOLDOUT_PATH = "Dataset/splitted/Hourly-train-holdout.csv"
TEST_HOLDOUT_PATH= "Dataset/splitted/Hourly-test-holdout.csv"

data_loader = M4DataLoader(TRAIN_PATH, TEST_PATH, TRAIN_HOLDOUT_PATH, TEST_HOLDOUT_PATH,
                           x_augmentations, y_augmentations, LOOKBACK, HORIZON)

train_x, train_y = data_loader.get_training_data()
test_x, test_y = data_loader.get_test_data()
holdout_x, holdout_y = data_loader.get_holdout_data()

training_data_generator = M4Generator(train_x, train_y, BATCH_SIZE, FEATURES_NUMBER)
test_data_generator = M4Generator(test_x, test_y, BATCH_SIZE, FEATURES_NUMBER)
holdout_data_generator = M4Generator(holdout_x, holdout_y, BATCH_SIZE, FEATURES_NUMBER)


model = M4Model(hidden_layer_size=HIDDEN_LAYER_SIZE, hidden_layers=HIDDEN_LAYERS,
                batch_size=BATCH_SIZE, lookback=LOOKBACK, 
                output_size=OUTPUT_SIZE, learning_rate=LEARNING_RATE, loss = LOSS,
                dropout_ratio = DROPOUT_RATIO, features_number = FEATURES_NUMBER, 
                clipvalue=CLIP_VALUE, callbacks = [], pi_params=PI_PARAMS)

model.train(training_data_generator, test_data_generator, epochs=EPOCHS)

MODEL_BASE_DIR = 'models/kl_divergance/2-LSTM'
model.save(MODEL_BASE_DIR)