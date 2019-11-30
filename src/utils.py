import pandas as pd
import numpy as np

from tensorflow.python.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import  ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from keras.models import model_from_json

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


def load_model(model_json_path, model_weights_path):
    
    # load json and create model
    json_file = open(model_json_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    
    # load weights into new model
    model.load_weights(model_weights_path)
    print("Loaded model from disk")
    
    return model

def save_model(model, model_json_path, model_weights_path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)
    
    # serialize weights to HDF5
    model.save_weights(model_weights_path)
    print("Saved model to disk")

def decompose_time_serie(data):
    serie_range = pd.date_range(freq="h", start=0, periods=data.shape[0])
    serie = pd.Series(data, index= serie_range)
    decomposition = seasonal_decompose(serie)

    return decomposition

def exponential_smoothing(data, forcasting_horizon=48, seasonal_periods=24, seasonal='add', trend='add'):
    model =  ExponentialSmoothing(data, seasonal_periods=seasonal_periods, seasonal=seasonal, trend='add')
    fitted_model = model.fit()

    return fitted_model.forecast(forcasting_horizon)

