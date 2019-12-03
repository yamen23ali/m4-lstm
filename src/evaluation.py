import sys
#sys.path.append('../src')

import numpy as np
import keras
import tensorflow as tf

from .error_functions import *
from keras.models import model_from_json

def reshape_data_in_batches(X, Y, batch_size):
    
    complete_batches = np.floor(X.shape[0]/batch_size)+1
    missing_samples = int((complete_batches*batch_size) - X.shape[0])
    
    X = np.concatenate((X, X[:missing_samples,:]), axis=0)
    Y = np.concatenate((Y, Y[:missing_samples,:]), axis=0)

    X = X.reshape(-1,batch_size, X.shape[1], 1)
    Y = Y.reshape(-1,batch_size, Y.shape[1],1)
    
    return X, Y

def evaluate_model(model, X, Y, error_function):
    
    X,Y = reshape_data_in_batches(X, Y, model.batch_size)
    
    errors = []
    
    for x_batch, y_batch in zip(X, Y):
        predictedY = model.predict(x_batch)
        errors.append(error_function(y_batch[:,:,0], predictedY))
        
    return np.mean(errors)

