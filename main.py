import sys
sys.path.append('src')

import numpy as np
import argparse, sys

from m4_generator import M4Generator
from error_functions import *
from visualization import *
from evaluation import evaluate_model
from m4_model import M4Model


#============= Get and Parse Arguments

parser=argparse.ArgumentParser()

parser.add_argument('--EPOCHS', help='Number of epochs')
parser.add_argument('--BATCH_SIZE', help='Batch Size')
parser.add_argument('--LOOKBACK', help='Look back window')
parser.add_argument('--HIDDEN_LAYER_SIZE', help='Hidden Layers Number')

args=parser.parse_args()


#=============== Define and Train Model

EPOCHS = int(args.EPOCHS)
BATCH_SIZE = int(args.BATCH_SIZE)
LOOKBACK = int(args.LOOKBACK)
HORIZON = 48
HIDDEN_LAYER_SIZE = int(args.HIDDEN_LAYER_SIZE)

model_name = f'LSTM_E{EPOCHS}_B{BATCH_SIZE}_H{HIDDEN_LAYER_SIZE}_L{LOOKBACK}'

gen = M4Generator("Dataset/Train/Hourly-train.csv", "Dataset/Test/Hourly-test.csv",
                  LOOKBACK, HORIZON, BATCH_SIZE)

model = M4Model(hidden_layer_size=HIDDEN_LAYER_SIZE, batch_size=BATCH_SIZE, lookback=LOOKBACK, 
        horizon=HORIZON)

hist = model.train(gen, epochs=EPOCHS)

model.save(f'models/{model_name}.json', f'models/{model_name}.h5')

#=================== Evaluate Model
train_x, train_y, test_x, test_y = gen.get_data()

train_error = evaluate_model(model, train_x, train_y, smapetf)
test_error = evaluate_model(model, test_x, test_y, smapetf)

with open(f'models/{model_name}.txt', 'a') as file:
    file.write(f'Training Error: {train_error}\n')
    file.write(f'Test Error: {test_error}')



