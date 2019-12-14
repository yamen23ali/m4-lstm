import sys

import numpy as np
import keras
import tensorflow as tf

from src.m4_evaluation_loss_functions import *
from src.m4_data_loader import M4DataLoader
from src.m4_model import M4Model
from src.utils import *
from src.data_augmentations import *

from keras.models import model_from_json

    
def evaluate_snaive(X, Y):
    predictedY = X[:,-Y.shape[1]:]
    
    return np.mean(mase(X, Y, predictedY))

def evaluate_naive(X, Y):
    predictedY =  np.tile(X[:,-1][:,np.newaxis], Y.shape[1])

    return np.mean(mase(X, Y, predictedY))

def evaluate_exponential_smoothing(X, Y, loss_function):
    errors = []
    for x,y in zip(X,Y):
        predictedY = exponential_smoothing(x)
        loss = mase(X, predictedY[:,np.newaxis], y[:,np.newaxis])
        errors = np.append(errors, loss)
    return np.mean(errors)

def evaluate_model(model, X, Y, loss_function):

    predictedY = model.predict(X)

    if model.features_number == 1: 
        X = X[:,:, np.newaxis]

    return loss_function(X[:,:,0], Y[:,:48], predictedY[:,:48]).numpy().mean()
    

def evaluate_model1(model, X, Y, loss_function):

    predictedY = model.predict(X)

    if loss_function == mase:
        if model.features_number == 1: X = X[:,:, np.newaxis]

        return mase(X[:,:,0], Y[:,:48], predictedY[:,:48]).numpy().mean()
    
    return loss_function(Y[:,:48], predictedY[:,:48]).numpy()


def load_and_evaluate_model(model_base_dir, training_data_dir, test_data_dir, x_augmentations, y_augmentations, loss_function):
    model = M4Model()
    hyperparameters = model.load(model_base_dir)

    x_augmentations = x_augmentations[:model.features_number-1]
    y_augmentations = y_augmentations[:model.features_number-1]

    data_loader = M4DataLoader(training_data_dir, test_data_dir, 
                           x_augmentations, 
                           y_augmentations,
                           model.lookback,  validation_ratio=0.05)


    test_x, test_y = data_loader.get_test_data()
    validate_x, validate_y = data_loader.get_validation_data()

    test_error = evaluate_model(model, test_x, test_y, loss_function).mean()

    if model.features_number == 1: test_x = test_x[:,:, np.newaxis]

    naive_test_error = evaluate_naive(test_x[:,:,0], test_y[:,:48])
    snaive_test_error = evaluate_snaive(test_x[:,:,0], test_y[:,:48])

    validation_error = evaluate_model(model, validate_x, validate_y, loss_function).mean()

    if model.features_number == 1: validate_x = validate_x[:,:, np.newaxis]
    
    naive_validation_error = evaluate_naive(validate_x[:,:,0], validate_y[:,:48])
    snaive_validation_error = evaluate_snaive(validate_x[:,:,0], validate_y[:,:48])


    return {
    'hyperparameters': hyperparameters,
    'test_error': round(test_error,3), 
    'validation_error': round(validation_error, 3),
    'naive_test_error': round(naive_test_error, 3), 
    'snaive_test_error': round(snaive_test_error, 3),
    'naive_validation_error': round(naive_validation_error, 3),
    'snaive_validation_error': round(snaive_validation_error, 3)
    }

def sort_by_prediction_error(model, X, Y, loss_function):

    predictedY = model.predict(X)

    errors = loss_function(Y[:,:48], predictedY[:,:48]).numpy()
    
    # Ascending sorting for serires based on prediction error
    sorted_errors_indx = errors.argsort()

    X = X[sorted_errors_indx,:]
    Y = Y[sorted_errors_indx,:]
    predictedY = predictedY[sorted_errors_indx, :]
    errors = errors[sorted_errors_indx]
    
    return X, Y, predictedY, errors