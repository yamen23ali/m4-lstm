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


def evaluate_combined_model(data_model, diff_model, X, Y, error_function):
    
    X_data,Y_data = reshape_data_in_batches(X[:,:,0], Y[:,:,0], model.batch_size)
    X_diff,Y_diff = reshape_data_in_batches(X[:,:,1], Y[:,:,1], model.batch_size)
    
    errors = []
    
    for x_data_batch, y_data_batch, x_diff_batch, y_diff_batch in zip(X_data, Y_data, X_diff, Y_diff):
        predictedData = data_model.predict(x_data_batch)
        predictedDiff = diff_model.predict(x_diff_batch)
        combinedPred = predictedData + predictedDiff
        combinedPred = combinedPred[:,:-1]
        combinedPred = np.hstack((predictedData[:,0][:,np.newaxis], combinedPred))
        final_pred = (predictedData + combinedPred)/2

        errors.append(error_function(y_data_batch[:,:,0], final_pred))
        
    return np.mean(errors)

