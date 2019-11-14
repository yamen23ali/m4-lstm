from tensorflow.python.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_top_autcorr(data_frame, max_lag = 500, highest_corr = 10):
    lag_corr = []
    for j in range(0,max_lag):
        corrs = []
        for i in range(0,data_frame.shape[0]):
            ts = pd.Series(data_frame.iloc[[i]].values[0])
            corrs.append(ts.autocorr(lag=j))
        lag_corr.append(np.mean(corrs))
    
    correlations = np.array(lag_corr)    
    highest_correlations_ind = np.flip(correlations.argsort()[-highest_corr:])
    return correlations[highest_correlations_ind], highest_correlations_ind

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
