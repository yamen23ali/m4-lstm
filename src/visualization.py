import numpy as np
import matplotlib.pyplot as plt

from .utils import read_raw_data

def plot_series(X, Y, predictedY, series_num = 3):
    """
        Plot multiple series on the same figure

        Args:
            X (array_like): The first part of multiple time-series (i.e. lookback)
            Y (array_like): The second part of multiple time-series (i.e. horizon)
            predictedY (array_like): The predicted second part of multiple time-series (i.e. predicted horizon)
            series_num (int): The number of series to plot
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    time_x= np.arange(0, X.shape[1])
    time_y= np.arange(X.shape[1], X.shape[1] + Y.shape[1])
    
    for i in range(0,series_num):
        axs[0].plot(time_x, X[i,:])
        axs[0].plot(time_y, Y[i,:])
        axs[1].plot(time_x, X[i,:])    
        axs[1].plot(time_y, predictedY[i,:])
        
    axs[0].set_title('True Values')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')
    axs[1].set_title('Point Predictions')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Value')
    axs[0].axvline(x=X.shape[1], color='r')
    axs[1].axvline(x=X.shape[1], color='r')


def plot_serie_and_prediction(X, Y, predictedY, serie_indx = 0, save_path = ''):
    """
        Plot a time-serie (lookback, horzion, prediction) and save the resulted figure

        Args:
            X (array_like): The first part of multiple time-series (i.e. lookback)
            Y (array_like): The second part of multiple time-series (i.e. horizon)
            predictedY (array_like): The predicted second part of multiple time-series (i.e. predicted horizon)
            serie_indx (int): The serie to plot
            save_path (str): The path to save the resulted figure in. If empty the result won't be saved.
    """
    fig, axs = plt.subplots(1, 1, figsize=(40, 20))
    
    true_series_time_axes = np.arange(0, X.shape[1] + Y.shape[1] )
    predicted_series_time_axes= np.arange(X.shape[1], X.shape[1] + Y.shape[1])
    
    true_series_value_axes = np.concatenate( (X, Y), axis = 1)
    
    axs.plot(true_series_time_axes, true_series_value_axes[serie_indx,:], color='b')
    axs.plot(predicted_series_time_axes, predictedY[serie_indx,:], color='r')
    axs.axvline(x=X.shape[1], color='g')
    
    axs.set_xlabel('Time', fontsize=20)
    axs.set_ylabel('Value', fontsize=20)    
    axs.set_title('True Values & Point Predictions', fontsize=20)
    axs.legend(['True Serie', 'Predicted Serie'], prop={'size': 30})

    if save_path !='':
        plt.savefig(save_path)

def plot_m4_complete_series(training_file_path, test_file_path):
    """
        Plot complete M4 time-series after joining their parts from (training, test)  files.

        Args:
            training_file_path (str): The training csv file path
            test_file_path (str): The test csv file path
    """

    training_data = read_raw_data(training_file_path)
    test_data = read_raw_data(test_file_path)
    
    for x, y in zip(training_data, test_data):
        x = x[~np.isnan(x), np.newaxis]
        ts = np.vstack((x, y[:,np.newaxis]))
        time = np.arange(0, ts.shape[0])

        fig, axs = plt.subplots(1, 1, figsize=(40, 20))
        
        axs.plot(time, ts, color='b')
        axs.axvline(x=48, color='g')
        plt.show()


def plot_serie_with_bounds(x, y, predictedY, lower_bound, upper_bound, serie_indx = 0, save_path = ''):
    """
        Plot a time-serie (lookback, horzion, prediction, lower bound, upper bound) and save the resulted figure

        Args:
            X (array_like): The first part of multiple time-series (i.e. lookback)
            Y (array_like): The second part of multiple time-series (i.e. horizon)
            predictedY (array_like): The predicted second part of multiple time-series (i.e. predicted horizon)
            lower_bound (array_like): The predicted lower bounds of multiple time-series.
            upper_bound (array_like): The predicted upper bounds of multiple time-series.
            serie_indx (int): The serie to plot
            save_path (str): The path to save the resulted figure in. If empty the result won't be saved.
    """
    
    fig, axs = plt.subplots(1, 1, figsize=(40, 20))
    
    true_series_time_axes = np.arange(0, x.shape[1] + y.shape[1])
    predicted_series_time_axes= np.arange(x.shape[1], x.shape[1] + y.shape[1])
    
    true_series_value_axes = np.concatenate((x, y), axis = 1)
    
    axs.plot(true_series_time_axes, true_series_value_axes[serie_indx,:], color='b')
    axs.plot(predicted_series_time_axes, predictedY[serie_indx,:], color='r')
    axs.plot(predicted_series_time_axes, lower_bound[serie_indx,:], color='y')
    axs.plot(predicted_series_time_axes, upper_bound[serie_indx,:], color='c')
    axs.axvline(x=x.shape[1], color='g')
    
    axs.set_xlabel('Time', fontsize=20)
    axs.set_ylabel('Value', fontsize=20)    
    axs.set_title('True Values & Point Predictions', fontsize=20)
    axs.legend(['True Serie', 'Predicted Serie'], prop={'size': 30})

    if save_path !='':
        plt.savefig(save_path)
