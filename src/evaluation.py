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
    """
        Generate a Snaive model points predictions and evaluate them against the M4_MASE error function

        Args:
            X (array_like): The insample data ( i.e. the data we have in hand that will be fed to the model)
            Y (array_like): The target data ( i.e. the actual data we hope to predict)

        Returns:
            float: The M4_MASE error on the Snaive predictions
    """
    predictedY = X[:,-Y.shape[1]:]
    
    return np.mean(m4_mase(X, Y, predictedY))

def evaluate_naive(X, Y):
    """
        Generate a Naive model points predictions and evaluate them against the M4_MASE error function

        Args:
            X (array_like): The insample data ( i.e. the data we have in hand that will be fed to the model)
            Y (array_like): The target data ( i.e. the actual data we hope to predict)

        Returns:
            float: The M4_MASE error on the Naive predictions
    """
    predictedY =  np.tile(X[:,-1][:,np.newaxis], Y.shape[1])

    return np.mean(m4_mase(X, Y, predictedY))

def evaluate_naive_PI(X, Y):
    """
        Generate a Naive model upper and lower bounds predictions and evaluate them against ACD and MSIS error functions

        Args:
            X (array_like): The insample data ( i.e. the data we have in hand that will be fed to the model)
            Y (array_like): The target data ( i.e. the actual data we hope to predict)

        Returns:
            (float, float): The ACD and MSIS errors on the predicted PIs (i.e. upper and lower bounds)
    """
    y = Y[:,:48]
    lower, upper = naive_pi(y)
    acd_err = acd( y[:,:-1], lower, upper)
    msis_err = msis(X[:,:],  y[:,:-1], lower, upper)

    return acd_err, msis_err

def evaluate_exponential_smoothing(X, Y, loss_function):
    """
        Generate an exponential smoothing points predictions and evaluate them against some loss function

        Args:
            X (array_like): The insample data ( i.e. the data we have in hand that will be fed to the model)
            Y (array_like): The target data ( i.e. the actual data we hope to predict)
            loss_function (func): The loss function to evaluate the predictions against

        Returns:
            float: The loss on the predicted points
    """
    errors = []
    for x,y in zip(X,Y):
        predictedY = exponential_smoothing(x)
        loss = m4_mase(X, predictedY[:,np.newaxis], y[:,np.newaxis])
        errors = np.append(errors, loss)
    return np.mean(errors)

def evaluate_model(model, X, Y, loss_function):
    """
        Get a specific model predictions and evaluate them against some loss function

        Args:
            model (:obj: `M4Model`): A trained M4Model to evaluate  its predictions
            X (array_like): The insample data ( i.e. the data we have in hand that will be fed to the model)
            Y (array_like): The target data ( i.e. the actual data we hope to predict)
            loss_function (func): The loss function to evaluate the predictions against

        Returns:
            float: The loss on the predicted points
    """

    predictedY = model.predict(X)

    if model.features_number == 1:
        X = X[:,:, np.newaxis]

    return loss_function(X[:,:,0], Y[:,:48], predictedY[:,:48]).numpy().mean()

def evaluate_berken_PI(model, X, Y):
    """
        Get a specific model lower and upper bounds predictions and evaluate them against ACD and MSIS error functions.
        The passed model should be trained on Berken's error function and outputs the data as defined in Berken's thesis.

        Args:
            model (:obj: `M4Model`): A trained M4Model to evaluate  its predictions
            X (array_like): The insample data ( i.e. the data we have in hand that will be fed to the model)
            Y (array_like): The target data ( i.e. the actual data we hope to predict)

        Returns:
            (float, float): The ACD and MSIS errors on the predicted PIs (i.e. upper and lower bounds)
    """

    predictedY = model.predict(X)

    lower_bound = predictedY[:,-48:]
    upper_bound = predictedY[:,48:-48]

    acd_err = acd(Y[:,:48], lower_bound, upper_bound)
    msis_err = msis(X, Y[:,:48], lower_bound, upper_bound)

    return acd_err, msis_err

def evaluate_kl_PI(model, X, Y):
    """
        Get a specific model lower and upper bounds predictions and evaluate them against ACD and MSIS error functions.
        The passed model should be trained on kl_divergence error function and outputs the data as defined in the KL divergence approach.

        Args:
            model (:obj: `M4Model`): A trained M4Model to evaluate  its predictions
            X (array_like): The insample data ( i.e. the data we have in hand that will be fed to the model)
            Y (array_like): The target data ( i.e. the actual data we hope to predict)

        Returns:
            (float, float): The ACD and MSIS errors on the predicted PIs (i.e. upper and lower bounds)
    """

    predictedY = model.predict(X)

    y = Y[:,:48]
    x = X[:,:,0]

    lower_bound = predictedY[:,:48] - 2*tf.abs(predictedY[:,-48:])
    upper_bound = predictedY[:,:48] + 2*tf.abs(predictedY[:,-48:])

    acd_err = acd(y, lower_bound.numpy(), upper_bound.numpy())
    msis_err = msis(x, y, lower_bound.numpy(), upper_bound.numpy())

    return acd_err, msis_err

def modify_augmentations(model, augmentations):

    """
        A helper function to modify augmentations based on trained model values

        Args:
            model (:obj: `M4Model`): A trained M4Model to evaluate  its predictions
            augmentations (:obj:`list` of Augmentations): A list of augmentations to modify

        Returns:
            (:obj:`list` of Augmentations): The augmentations after being modified with appropriate values from a previously trained model
    """
    
    augmentations = augmentations[:model.features_number-1]
    
    for augmentation in augmentations:
        if isinstance(augmentation, StdAugmentation):
            augmentation.set_pi_params(model.pi_params)

    return augmentations

def load_and_evaluate_model(model_base_dir, train_path, test_path,
    train_holdout_path, test_holdout_path, x_augmentations, y_augmentations, loss_function):
    """
        Load a pretrained model and evaluate its point prediction against a specific loss function

        Args:
            model_base_dir (str): The path of the saved model in the file system
            train_path (str): The path of the training.csv file of the training_test dataset
            test_path (str): The path of the test.csv file of the training_test dataset
            train_holdout_path (str): The path of the training.csv file of the holdout dataset
            test_holdout_path (str): The path of the test.csv file of the holdout dataset
            x_augmentations (:obj:`list` of Augmentations): A list of augmentations to apply on the input data (i.e. data fed to the model)
            y_augmentations (:obj:`list` of Augmentations): A list of augmentations to apply on the output data (i.e. predictions of the model)
            loss_function (func): The loss function to evaluate the predictions against

        Returns:
            (dict): TThe loaded model hyperparameters and the model, snaive and naive error on the holdout and test data
    """
    
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
    train_holdout_path, test_holdout_path,x_augmentations, y_augmentations, loss_function):
    """
        Load a pretrained model and evaluate its lower and upper bounds predictions against a specific loss function

        Args:
            model_base_dir (str): The path of the saved model in the file system
            train_path (str): The path of the training.csv file of the training_test dataset
            test_path (str): The path of the test.csv file of the training_test dataset
            train_holdout_path (str): The path of the training.csv file of the holdout dataset
            test_holdout_path (str): The path of the test.csv file of the holdout dataset
            x_augmentations (:obj:`list` of Augmentations): A list of augmentations to apply on the input data (i.e. data fed to the model)
            y_augmentations (:obj:`list` of Augmentations): A list of augmentations to apply on the output data (i.e. predictions of the model)
            loss_function (func): The loss function to evaluate the predictions against

        Returns:
            (dict): The loaded model hyperparameters and the model, snaive and naive error on the holdout and test data
    """
    model = M4Model()
    hyperparameters = model.load(model_base_dir)

    # Modify augmentations based on trained model values
    x_augmentations = modify_augmentations(model, x_augmentations)
    y_augmentations = modify_augmentations(model, y_augmentations)

    data_loader = M4DataLoader(train_path, test_path, train_holdout_path, test_holdout_path,
        x_augmentations, y_augmentations, model.lookback)


    test_x, test_y = data_loader.get_test_data()
    holdout_x, holdout_y = data_loader.get_holdout_data()

    acd_test, msis_test = loss_function(model, test_x, test_y)

    if model.features_number == 1: test_x = test_x[:,:, np.newaxis]

    acd_naive_test, msis_naive_test = evaluate_naive_PI(test_x[:,:,0], test_y[:,:48])

    acd_holdout, msis_holdout = loss_function(model, holdout_x, holdout_y)

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
    """
        Sort the model predictions and the associated data by the error.
        The main idea of this function is to get the (input data, predictions) sorted based on the ascending error value.

        Args:
            model (:obj: `M4Model`): A trained M4Model to evaluate  its predictions
            X (array_like): The insample data ( i.e. the data we have in hand that will be fed to the model)
            Y (array_like): The target data ( i.e. the actual data we hope to predict)
            loss_function (func): The loss function to evaluate the predictions against

        Returns:
            (array_like, array_like, array_like, array_like): (The sorted input data, The sorted ground truth data, The sorted predictions, THe sorted errors)
    """

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
    """
        Get model predictions on a specific dataset and save the results.
        Since a standarization was applied on the data before training, we need to unstandarize the output of the model.

        Args:
            model_dir (str): A path to save the model under it
            data_loader (:obj: `M4DataLoader`): A data loader instance that will give the model input data
            horizon (int): The prediction horizon (i.e. how many steps a head to predict)
    """
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

    os.mkdir(f'{model_dir}/Test')
    np.savetxt(f'{model_dir}/Test/point.csv', unstandarized_point[:test_x.shape[0]], delimiter=",")
    np.savetxt(f'{model_dir}/Test/lower.csv', unstandarized_lower[:test_x.shape[0]], delimiter=",")
    np.savetxt(f'{model_dir}/Test/upper.csv', unstandarized_upper[:test_x.shape[0]], delimiter=",")

    os.mkdir(f'{model_dir}/Holdout')
    np.savetxt(f'{model_dir}/Holdout/point.csv', unstandarized_point[test_x.shape[0]:], delimiter=",")
    np.savetxt(f'{model_dir}/Holdout/lower.csv', unstandarized_lower[test_x.shape[0]:], delimiter=",")
    np.savetxt(f'{model_dir}/Holdout/upper.csv', unstandarized_upper[test_x.shape[0]:], delimiter=",")
