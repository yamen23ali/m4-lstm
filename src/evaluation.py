import sys

import numpy as np
import keras
import tensorflow as tf

from src.error_functions import *
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

def sort_by_prediction_error(model, X, Y, error_function):
    predictions = np.empty(shape=[0, Y.shape[1]])
    X,Y = reshape_data_in_batches(X, Y, model.batch_size)

    errors = np.empty(shape=[0, 0])
    
    for batch_x, batch_y in zip(X, Y):

        batch_predictions = model.predict(batch_x)

        predictions = np.concatenate((predictions, batch_predictions), axis = 0)

        batch_errors = error_function(batch_y[:,:,0], batch_predictions)

        errors = np.append(errors, batch_errors)
        
    X = X.reshape(-1, X.shape[2])
    Y = Y.reshape(-1, Y.shape[2])
    
    # Ascending sorting for serires based on prediction error
    sorted_errors_indx = errors.argsort()
    X = X[sorted_errors_indx]
    Y = Y[sorted_errors_indx]
    predictions = predictions[sorted_errors_indx]
    errors = errors[sorted_errors_indx]
    
    return X, Y, predictions, errors