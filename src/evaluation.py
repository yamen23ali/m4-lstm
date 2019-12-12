import sys

import numpy as np
import keras
import tensorflow as tf

from src.error_functions import *
from src.m4_data_loader import M4DataLoader
from src.m4_model import M4Model
from src.utils import *
from src.data_augmentations import *

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

def evaluate_model(model, X, Y, loss_function):

    predictedY = model.predict(X)
    return loss_function(Y[:,:48], predictedY[:,:48]).numpy().mean()


def load_and_evaluate_model(model_base_dir, training_data_dir, test_data_dir, loss_function):
    model = M4Model()
    hyperparameters = model.load(model_base_dir)

    stdAugmentation = StdAugmentation(model.pi_params)
    diffAugmentation = DiffAugmentation()
    x_augmentations = [stdAugmentation]
    y_augmentations = [stdAugmentation]

    data_loader = M4DataLoader(training_data_dir, test_data_dir, 
                           x_augmentations, 
                           y_augmentations,
                           model.lookback,  validation_ratio=0.05)


    train_x, train_y = data_loader.get_training_data()
    test_x, test_y = data_loader.get_test_data()
    validate_x, validate_y = data_loader.get_validation_data()


    training_error = evaluate_model(model, train_x, train_y, loss_function)
    
    test_error = evaluate_model(model, test_x, test_y, loss_function)
    
    validation_error = evaluate_model(model, validate_x, validate_y, loss_function)

    return hyperparameters, round(training_error, 3), round(test_error,3), round(validation_error, 3)

def sort_by_prediction_error(model, X, Y, loss_function):

    predictedY = model.predict(X)

    errors = loss_function(Y[:,:48], predictedY[:,:48]).numpy()[:,0]
    
    # Ascending sorting for serires based on prediction error
    sorted_errors_indx = errors.argsort()

    X = X[sorted_errors_indx,:,:]
    Y = Y[sorted_errors_indx,:]
    predictedY = predictedY[sorted_errors_indx, :]
    errors = errors[sorted_errors_indx]
    
    return X, Y, predictedY, errors