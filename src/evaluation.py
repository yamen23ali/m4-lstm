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
    
    return np.mean(m4_mase(X, Y, predictedY))

def evaluate_naive(X, Y):
    predictedY =  np.tile(X[:,-1][:,np.newaxis], Y.shape[1])

    return np.mean(m4_mase(X, Y, predictedY))

def evaluate_naive_PI(X, Y):
    y = Y[:,:48]
    lower, upper = naive_pi(y)
    acd_err = acd( y[:,:-1], lower, upper)
    msis_err = msis(X[:,:],  y[:,:-1], lower, upper)

    return acd_err, msis_err

def evaluate_exponential_smoothing(X, Y, loss_function):
    errors = []
    for x,y in zip(X,Y):
        predictedY = exponential_smoothing(x)
        loss = m4_mase(X, predictedY[:,np.newaxis], y[:,np.newaxis])
        errors = np.append(errors, loss)
    return np.mean(errors)

def evaluate_model(model, X, Y, loss_function):

    predictedY = model.predict(X)

    if model.features_number == 1:
        X = X[:,:, np.newaxis]

    return loss_function(X[:,:,0], Y[:,:48], predictedY[:,:48]).numpy().mean()

def evaluate_berken_PI(model, X, Y):

    predictedY = model.predict(X)

    lower_bound = predictedY[:,-48:]
    upper_bound = predictedY[:,48:-48]

    acd_err = acd(Y[:,:48], lower_bound, upper_bound)
    msis_err = msis(X, Y[:,:48], lower_bound, upper_bound)

    return acd_err, msis_err

def evaluate_kl_PI(model, X, Y):

    predictedY = model.predict(X)

    y = Y[:,:48]
    x = X[:,:,0]

    lower_bound = predictedY[:,:48] - 2*tf.abs(predictedY[:,-48:])
    upper_bound = predictedY[:,:48] + 2*tf.abs(predictedY[:,-48:])

    acd_err = acd(y, lower_bound.numpy(), upper_bound.numpy())
    msis_err = msis(x, y, lower_bound.numpy(), upper_bound.numpy())

    return acd_err, msis_err

def modify_augmentations(model, augmentations):
    
    augmentations = augmentations[:model.features_number-1]
    
    for augmentation in augmentations:
        if isinstance(augmentation, StdAugmentation):
            augmentation.set_pi_params(model.pi_params)

    return augmentations

def load_and_evaluate_model(model_base_dir, train_path, test_path,
    train_holdout_path, test_holdout_path,x_augmentations, y_augmentations, loss_function):
    
    model = M4Model()
    hyperparameters = model.load(model_base_dir)

    # Modify augmentations based on trained model values
    x_augmentations = modify_augmentations(model, x_augmentations)
    y_augmentations = modify_augmentations(model, y_augmentations)

    data_loader = M4DataLoader(train_path, test_path, train_holdout_path, test_holdout_path,
        x_augmentations, y_augmentations, model.lookback)


    test_x, test_y = data_loader.get_test_data()
    holdout_x, holdout_y = data_loader.get_holdout_data()

    test_error = evaluate_model(model, test_x, test_y, loss_function).mean()

    if model.features_number == 1: test_x = test_x[:,:, np.newaxis]

    naive_test_error = evaluate_naive(test_x[:,:,0], test_y[:,:48])
    snaive_test_error = evaluate_snaive(test_x[:,:,0], test_y[:,:48])

    holdout_error = evaluate_model(model, holdout_x, holdout_y, loss_function).mean()

    if model.features_number == 1: holdout_x = holdout_x[:,:, np.newaxis]
    
    naive_holdout_error = evaluate_naive(holdout_x[:,:,0], holdout_y[:,:48])
    snaive_holdout_error = evaluate_snaive(holdout_x[:,:,0], holdout_y[:,:48])


    return {
    'hyperparameters': hyperparameters,
    'test_error': round(test_error,3), 
    'holdout_error': round(holdout_error, 3),
    'naive_test_error': round(naive_test_error, 3), 
    'snaive_test_error': round(snaive_test_error, 3),
    'naive_holdout_error': round(naive_holdout_error, 3),
    'snaive_holdout_error': round(snaive_holdout_error, 3)
    }


def load_and_evaluate_model_PI(model_base_dir, train_path, test_path,
    train_holdout_path, test_holdout_path,x_augmentations, y_augmentations, evaluation_function):
    
    model = M4Model()
    hyperparameters = model.load(model_base_dir)

    # Modify augmentations based on trained model values
    x_augmentations = modify_augmentations(model, x_augmentations)
    y_augmentations = modify_augmentations(model, y_augmentations)

    data_loader = M4DataLoader(train_path, test_path, train_holdout_path, test_holdout_path,
        x_augmentations, y_augmentations, model.lookback)


    test_x, test_y = data_loader.get_test_data()
    holdout_x, holdout_y = data_loader.get_holdout_data()

    acd_test, msis_test = evaluation_function(model, test_x, test_y)

    if model.features_number == 1: test_x = test_x[:,:, np.newaxis]

    acd_naive_test, msis_naive_test = evaluate_naive_PI(test_x[:,:,0], test_y[:,:48])

    acd_holdout, msis_holdout = evaluation_function(model, holdout_x, holdout_y)

    if model.features_number == 1: holdout_x = holdout_x[:,:, np.newaxis]
    
    acd_naive_holdout, msis_naive_holdout = evaluate_naive_PI(holdout_x[:,:,0], holdout_y[:,:48])

    return {
    'hyperparameters': hyperparameters,
    'acd_test': round(acd_test,3), 
    'acd_holdout': round(acd_holdout, 3),
    'acd_naive_test': round(acd_naive_test, 3), 
    'acd_naive_holdout': round(acd_naive_holdout, 3),

    'msis_test': round(msis_test,3), 
    'msis_holdout': round(msis_holdout, 3),
    'msis_naive_test': round(msis_naive_test, 3), 
    'msis_naive_holdout': round(msis_naive_holdout, 3),
    }

def sort_by_prediction_error(model, X, Y, loss_function):

    predictedY = model.predict(X)

    errors = loss_function(Y[:,:48], predictedY[:,:48]).numpy().flatten()
    # Ascending sorting for serires based on prediction error
    sorted_errors_indx = errors.argsort()

    X = X[sorted_errors_indx,:]
    Y = Y[sorted_errors_indx,:]
    predictedY = predictedY[sorted_errors_indx, :]
    errors = errors[sorted_errors_indx]
    
    return X, Y, predictedY, errors

def predict_and_save(model_dir, data_loader, horizon):
    model = M4Model()
    hyperparameters = model.load(model_dir)
    test_x, test_y = data_loader.get_test_data()
    holdout_x, holdout_y = data_loader.get_holdout_data()


    predictedY = model.predict(test_x)
    point_test = predictedY[:,:horizon]
    lower_bound_test = predictedY[:,:horizon] - 2*tf.abs(predictedY[:,-horizon:])
    upper_bound_test = predictedY[:,:horizon] + 2*tf.abs(predictedY[:,-horizon:])

    predictedY = model.predict(holdout_x)
    point_holdout = predictedY[:,:horizon]
    lower_bound_holdout = predictedY[:,:horizon] - 2*tf.abs(predictedY[:,-horizon:])
    upper_bound_holdout = predictedY[:,:horizon] + 2*tf.abs(predictedY[:,-horizon:])

    point = np.append(point_test, point_holdout, axis=0)
    unstandarized_point = data_loader.unstandarize_predictions(point)

    lower_bound = np.append(lower_bound_test, lower_bound_holdout, axis=0)
    unstandarized_lower = data_loader.unstandarize_predictions(lower_bound)

    upper_bound = np.append(upper_bound_test, upper_bound_holdout, axis=0)
    unstandarized_upper = data_loader.unstandarize_predictions(upper_bound)

    np.savetxt(f'{model_dir}/point_test.csv', unstandarized_point[:test_x.shape[0]], delimiter=",")
    np.savetxt(f'{model_dir}/lower_test.csv', unstandarized_lower[:test_x.shape[0]], delimiter=",")
    np.savetxt(f'{model_dir}/upper_test.csv', unstandarized_upper[:test_x.shape[0]], delimiter=",")

    np.savetxt(f'{model_dir}/point_holdout.csv', unstandarized_point[test_x.shape[0]:], delimiter=",")
    np.savetxt(f'{model_dir}/lower_holdout.csv', unstandarized_lower[test_x.shape[0]:], delimiter=",")
    np.savetxt(f'{model_dir}/upper_holdout.csv', unstandarized_upper[test_x.shape[0]:], delimiter=",")
