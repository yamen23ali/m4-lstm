import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K


def s_naive_error(X, freq= 24):
    """
        Calculate the error of a Snaive prediction on data points

        Args:
            X (array_like): The data to compute the error of Snaive prediction on
            freq (int): The frequency ( i.e. seasonality) of the data. Here it is 24 hours

        Returns:
            (float): The Snaive predicitons error
    """
    n = X.shape[1]
    X1 = X[:,:n-freq]
    X2 = X[:,freq:]
    
    return 1/(n - freq) * np.abs(X1 - X2).sum(axis=1)

def m4_mase(in_sample, yTrue, yPred):
    """
        The modified MASE error as suggested by M4 competition organizers

        Args:
            in_sample (array_like): The insample data ( i.e. model input)
            yTrue (array_like): The actual timeseries points of the horizon
            yPred (array_like): The predicted timeseries points, upper bounds and lower bounds of the horizon

        Returns:
            (float): The M4 MASE error of the predictions
    """
    naive_err = s_naive_error(in_sample)[:,np.newaxis]
    naive_err[naive_err == 0.0] = 0.001 # Just to avoid getting inf

    err = tf.abs(yTrue - yPred) / naive_err
    
    return tf.reduce_mean(err, axis=1)


def acd(yTrue, yLower, yUpper):
    """
        The ACD error of the upper and lower bounds predictions

        Args:
            yTrue (array_like): The actual timeseries points of the horizon
            yLower (array_like): The predicted lower bounds
            yUpper (array_like): The predicted upper bounds

        Returns:
            (float): The ACD error of the upper and lower bounds predictions
    """
    covered_from_lower = yTrue >= yLower
    covered_from_upper = yTrue <= yUpper
    covered = covered_from_lower & covered_from_upper
    
    return abs( covered.sum() / (yTrue.shape[0]*yTrue.shape[1]) - 0.95)

def msis(insample, yTrue, yLower, yUpper):
    """
        The ACD error of the upper and lower bounds predictions

        Args:
            in_sample (array_like): The insample data ( i.e. model input)
            yTrue (array_like): The actual timeseries points of the horizon
            yLower (array_like): The predicted lower bounds
            yUpper (array_like): The predicted upper bounds

        Returns:
            (float): The MSIS error of the upper and lower bounds predictions
    """
    
    ts_naive_err = s_naive_error(insample)
    
    coff = 40
    
    total_penalty = []
    
    for ts, tsUpper, tsLower, naive_err in zip(yTrue, yUpper, yLower, ts_naive_err):
        interval_width = tsUpper - tsLower
        
        missed_lower_indx = ts < tsLower
        missed_lower_penalty = coff*(tsLower[missed_lower_indx] - ts[missed_lower_indx])
        
        missed_upper_indx = ts > tsUpper
        missed_upper_penalty = coff*(ts[missed_upper_indx] - tsUpper[missed_upper_indx])
        
        ts_penalty = interval_width.sum() + missed_lower_penalty.sum() + missed_upper_penalty.sum()

        ts_penalty = ts_penalty / (ts.shape[0] * naive_err)
        
        total_penalty.append(ts_penalty)
    
    return np.array(total_penalty).mean()




