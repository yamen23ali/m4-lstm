import sys

import numpy as np
import keras
import tensorflow as tf

from src.error_functions import *
from src.m4_data_loader import M4DataLoader
from src.m4_model import M4Model
from src.utils import *

from keras.models import model_from_json


def evaluate_snaive(X, Y, loss_function):
    predictedY = X[:,-Y.shape[1]:]
    
    return np.mean(loss_function(Y, predictedY))

def evaluate_naive(X, Y, loss_function):
    predictedY = np.tile(X[:,-1][:,np.newaxis], Y.shape[1])

    return np.mean(loss_function(Y, predictedY))

def evaluate_exponential_smoothing(X, Y, loss_function):
    errors = []
    for x,y in zip(X,Y):
        predictedY = exponential_smoothing(x)
        loss = loss_function(predictedY[:,np.newaxis], y[:,np.newaxis])
        errors = np.append(errors, loss)
    return np.mean(errors)

def reshape_data_in_batches(X, Y, batch_size):
    
    complete_batches = np.floor(X.shape[0]/batch_size)+1
    missing_samples = int((complete_batches*batch_size) - X.shape[0])
    
    X = np.concatenate((X, X[:missing_samples,:]), axis=0)
    Y = np.concatenate((Y, Y[:missing_samples,:]), axis=0)

    X = X.reshape(-1,batch_size, X.shape[1], X.shape[2])
    Y = Y.reshape(-1,batch_size, Y.shape[1], 1)
    
    return X, Y

def evaluate_model(model, X, Y, loss_function):
    
    X,Y = reshape_data_in_batches(X, Y, model.batch_size)
    
    errors = []
    
    for x_batch, y_batch in zip(X, Y):
        predictedY = model.predict(x_batch)
        errors.append(loss_function(y_batch[:,:,0], predictedY))
        
    return np.mean(errors)

def load_and_evaluate_model(model_base_dir, training_data_dir, test_data_dir, loss_function):
    model = M4Model()
    hyperparameters = model.load(model_base_dir)

    data_loader = M4DataLoader(training_data_dir, test_data_dir, hyperparameters['lookback'])

    train_x, train_y = data_loader.get_training_data()
    test_x, test_y = data_loader.get_test_data()
    validate_x, validate_y = data_loader.get_validation_data()

    training_error = evaluate_model(model, train_x, train_y, loss_function)
    test_error = evaluate_model(model, test_x, test_y, loss_function)
    validation_error = evaluate_model(model, validate_x, validate_y, loss_function)

    return round(training_error, 4), round(test_error,4), round(validation_error, 2)

def sort_by_prediction_error(model, X, Y, error_function):
    predictions = np.empty(shape=[0, Y.shape[1]])
    X,Y = reshape_data_in_batches(X, Y, model.batch_size)

    errors = np.empty(shape=[0, 0])
    
    for batch_x, batch_y in zip(X, Y):

        batch_predictions = model.predict(batch_x)

        predictions = np.concatenate((predictions, batch_predictions), axis = 0)

        batch_errors = error_function(batch_y[:,:48,0], batch_predictions[:,:48])

        errors = np.append(errors, batch_errors)

    X = X[:,:,:,0]
    Y = Y[:,:,:48]

    X = X.reshape(-1, X.shape[2])
    Y = Y.reshape(-1, Y.shape[2])
    
    # Ascending sorting for serires based on prediction error
    sorted_errors_indx = errors.argsort()

    X = X[sorted_errors_indx]
    Y = Y[sorted_errors_indx]
    predictions = predictions[sorted_errors_indx]
    errors = errors[sorted_errors_indx]
    
    return X, Y, predictions, errors