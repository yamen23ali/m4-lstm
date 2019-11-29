import numpy as np
import matplotlib.pyplot as plt


def plot_series(x, y, predictedY, series_num = 3):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    time_x= np.arange(0, x.shape[1])
    time_y= np.arange(x.shape[1], x.shape[1] + y.shape[1])
    
    for i in range(0,series_num):
        axs[0].plot(time_x, x[i,:])
        axs[0].plot(time_y, y[i,:])
        axs[1].plot(time_x, x[i,:])    
        axs[1].plot(time_y, predictedY[i,:])
        
    axs[0].set_title('True Values')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')
    axs[1].set_title('Point Predictions')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Value')
    axs[0].axvline(x=x.shape[1], color='r')
    axs[1].axvline(x=x.shape[1], color='r')


def plot_serie_and_prediction(x, y, predictedY, serie_indx = 0):
    fig, axs = plt.subplots(1, 1, figsize=(40, 20))
    
    true_series_time_axes = np.arange(0, x.shape[1] + y.shape[1] )
    predicted_series_time_axes= np.arange(x.shape[1], x.shape[1] + y.shape[1])
    
    true_series_value_axes = np.concatenate( (x, y), axis = 1)
    
    axs.plot(true_series_time_axes, true_series_value_axes[serie_indx,:], color='b')
    axs.plot(predicted_series_time_axes, predictedY[serie_indx,:], color='r')
    axs.axvline(x=x.shape[1], color='g')
    
    axs.set_xlabel('Time', fontsize=20)
    axs.set_ylabel('Value', fontsize=20)    
    axs.set_title('True Values & Point Predictions', fontsize=20)
    axs.legend(['True Serie', 'Predicted Serie'], prop={'size': 30})

