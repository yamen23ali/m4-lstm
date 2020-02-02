import pandas as pd
import numpy as np
import os

from glob import glob
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import  ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose


def read_raw_data(file_path):
    df = pd.read_csv(file_path)

    del df['V1']
    
    return df.values

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

def create_model_dir(base_dir):

    models_numbers =[ int(dir_name.split('/')[-1]) for dir_name in glob(f'{base_dir}/*')]
    models_numbers.sort()
    
    if len(models_numbers) == 0: models_numbers = [1] 

    model_dir = f'{base_dir}/{models_numbers[-1] + 1}'
    os.mkdir(model_dir)

    return model_dir

def decompose_time_serie(data):
    serie_range = pd.date_range(freq="h", start=0, periods=data.shape[0])
    serie = pd.Series(data, index= serie_range)
    decomposition = seasonal_decompose(serie)

    return decomposition

def exponential_smoothing(data, forcasting_horizon=48, seasonal_periods=24, seasonal='add', trend='add'):
    model =  ExponentialSmoothing(data, seasonal_periods=seasonal_periods, seasonal=seasonal, trend='add')
    fitted_model = model.fit()

    return fitted_model.forecast(forcasting_horizon)

def naive_pi(data):
    predictions = data[:,1:]
    predictions_std = np.std(predictions, axis = 1)
    intervals = (predictions_std*1.96)[:,np.newaxis]
    upper = predictions + intervals
    lower = predictions - intervals

    return lower, upper


