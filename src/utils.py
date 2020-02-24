import pandas as pd
import numpy as np
import os

from glob import glob
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import  ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose


def read_raw_data(file_path):
    """
        Read the raw M4 data from csv file

        Args:
            file_path (str): The path of the csv file

        Returns:
            (array_like): The csv file content
    """
    df = pd.read_csv(file_path)

    del df['V1']
    
    return df.values

def create_model_dir(base_dir):
    """
        Create a directory to save the model in. This function will automatically name the directory in a proper way.

        Args:
            base_dir (str): The path where we want to create the directory
    """

    models_numbers =[ int(dir_name.split('/')[-1]) for dir_name in glob(f'{base_dir}/*')]
    models_numbers.sort()
    
    if len(models_numbers) == 0: models_numbers = [1] 

    model_dir = f'{base_dir}/{models_numbers[-1] + 1}'
    os.mkdir(model_dir)

    return model_dir

def decompose_time_serie(data):
    """
        Decompose the time-serie into trend, seasonality and resid components

        Args:
            data (array_like): The path of the csv file

        Returns:
            (:obj:`DecomposeResult`) : A object with seasonal, trend, and resid attributes.
    """
    serie_range = pd.date_range(freq="h", start=0, periods=data.shape[0])
    serie = pd.Series(data, index= serie_range)
    decomposition = seasonal_decompose(serie)

    return decomposition

def exponential_smoothing(data, forcasting_horizon=48, seasonal_periods=24, seasonal='add', trend='add'):
    """
        Apply exponential smoothing forecast for a given time-serie

        Args:
            data (array_like): The time-serie we want to forecast its future values
            forcasting_horizon (int): The number of future time steps to forecaset
            seasonal_periods (int): The seasonality value in the time-serie
            seasonal (str): The type of seasonality. {"add", "mul", "additive", "multiplicative", None}
            trend (str): The type of trend. {"add", "mul", "additive", "multiplicative", None}

        Returns:
            (array_like) : The forecasted values
    """
    model =  ExponentialSmoothing(data, seasonal_periods=seasonal_periods, seasonal=seasonal, trend=trend)
    fitted_model = model.fit()

    return fitted_model.forecast(forcasting_horizon)

def naive_pi(data):
    """
        Calculate the prediction intervals of a Naive model

        Args:
            data (array_like): The time-serie we want to calculate its predcition intervals

        Returns:
            (array_like, array_like) : The upper and lower bounds as predicted by a naive model
    """
    predictions = data[:,1:]
    predictions_std = np.std(predictions, axis = 1)
    intervals = (predictions_std*1.96)[:,np.newaxis]
    upper = predictions + intervals
    lower = predictions - intervals

    return lower, upper


