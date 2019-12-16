import sys
#sys.path.append('src')

import numpy as np
import argparse, sys
import src.error_functions as ef
import keras

from src.m4_data_loader import M4DataLoader
from src.m4_generator import M4Generator
from src.error_functions import *
from src.visualization import *
from src.evaluation import evaluate_model
from src.m4_model import M4Model

#============= Prepare 
keras.backend.set_floatx('float64')

#============= Get and Parse Arguments

parser=argparse.ArgumentParser()

parser.add_argument('--EPOCHS', help='Number of epochs')
parser.add_argument('--BATCH_SIZE', help='Batch Size')
parser.add_argument('--LOOKBACK', help='Look back window')
parser.add_argument('--HIDDEN_LAYER_SIZE', help='Hidden Layers Number')
parser.add_argument('--LOSS_FUNCTION', help='Loss Function')
parser.add_argument('--DROPOUT_RATIO', help='Dropout Ratio')

args=parser.parse_args()


#=============== Setup Hyperparameters

EPOCHS = int(args.EPOCHS)
BATCH_SIZE = int(args.BATCH_SIZE)
LOOKBACK = int(args.LOOKBACK)
HORIZON = 48
HIDDEN_LAYER_SIZE = int(args.HIDDEN_LAYER_SIZE)
LOSS_FUNCTION = args.LOSS_FUNCTION
DROPOUT_RATIO = float(args.DROPOUT_RATIO)

try:
	LOSS_FUNCTION = getattr(ef, LOSS_FUNCTION)
except Exception as e:
	print('Not defined in our loss functions, hopefully Keras has it !')


#=============== Load Data
data_loader = M4DataLoader("Dataset/Train/Hourly-train.csv", "Dataset/Test/Hourly-test.csv",
                  LOOKBACK, HORIZON, holdout_ratio=0.05)

train_x, train_y = data_loader.get_training_data()
training_data_generator = M4Generator(train_x, train_y, BATCH_SIZE)

test_x, test_y = data_loader.get_test_data()
test_data_generator = M4Generator(test_x, test_y, BATCH_SIZE)

validate_x, validate_y = data_loader.get_holdout_data()
holdout_data_generator = M4Generator(validate_x, validate_y, BATCH_SIZE)

#=============== Define and Train Model

model = M4Model(hidden_layer_size=HIDDEN_LAYER_SIZE, batch_size=BATCH_SIZE, lookback=LOOKBACK, 
        horizon=HORIZON, loss=LOSS_FUNCTION, dropout_ratio=DROPOUT_RATIO)

model.train(training_data_generator, test_data_generator, epochs=EPOCHS)

evaluation_loss = model.evaluate(holdout_data_generator)

print(f'Evaluation loss is : {evaluation_loss}')

model.save('models')
