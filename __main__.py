import sys
#sys.path.append('src')

import numpy as np
import argparse, sys
import src.error_functions as ef
import keras

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


#=============== Define and Train Model
gen = M4Generator("Dataset/Train/Hourly-train.csv", "Dataset/Test/Hourly-test.csv",
                  LOOKBACK, HORIZON, BATCH_SIZE)

model = M4Model(hidden_layer_size=HIDDEN_LAYER_SIZE, batch_size=BATCH_SIZE, lookback=LOOKBACK, 
        horizon=HORIZON, loss=LOSS_FUNCTION, dropout_ratio=DROPOUT_RATIO)

hist = model.train(gen, epochs=EPOCHS)

model_name = f'LSTM_E{EPOCHS}_B{BATCH_SIZE}_H{HIDDEN_LAYER_SIZE}_L{LOOKBACK}_ERR{args.LOSS_FUNCTION}_D{DROPOUT_RATIO}'

model.save(f'models/{model_name}.json', f'models/{model_name}.h5')

#=================== Evaluate Model
#train_x, train_y, test_x, test_y = gen.get_data()

#train_error = evaluate_model(model, train_x, train_y, smapetf)
#test_error = evaluate_model(model, test_x, test_y, smapetf)

#with open(f'models/{model_name}.txt', 'a') as file:
#    file.write(f'Training Error: {train_error}\n')
#    file.write(f'Test Error: {test_error}')



