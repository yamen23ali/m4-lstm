import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K

def __kl_div(mean_true, mean_pred, std_true, std_pred):
	d = std_true**2 + (mean_true - mean_pred)**2
	a = 2 * std_pred**2
	kl = K.log(std_pred / std_true) + (d/a) - 0.5

	return tf.reduce_mean(kl, axis = 1)[:,np.newaxis]


def kl_divergance(yTrue, yPred):
	tsTrue = yTrue[:,:48]
	tsPred = yPred[:,:48]

	stdTrue = yTrue[:,48:] 
	stdPred = tf.abs(yPred[:,-48:])

	return  __kl_div(tsTrue, tsPred, stdTrue, stdPred)

def naive_error(yTrue):
	return tf.reduce_mean(tf.abs(yTrue[:,1:] - yTrue[:,:-1]), axis=1)

def mase(yTrue, yPred):
	yTrue = yTrue[:,:48]
	naive_err = naive_error(yTrue)
	return mae(yTrue, yPred)[:,np.newaxis] / naive_err[:,np.newaxis]

def mae(yTrue, yPred):
	return tf.reduce_mean(tf.abs(yTrue - yPred), axis=1)

def rmse(yTrue, yPred):
	return tf.sqrt( tf.reduce_mean(tf.square(yTrue - yPred), axis=1))

def smape(yTrue, yPred):
    ratio = tf.abs(yTrue - yPred) / (tf.abs(yTrue) + tf.abs(yPred))
    return 200/yPred.shape[1] * tf.reduce_sum(ratio, axis = 1)

def acd(yTrue, yLower, yUpper):
    covered_from_lower = yTrue >= yLower
    covered_from_upper = yTrue <= yUpper
    covered = covered_from_lower & covered_from_upper
    
    return abs( covered.sum() / (yTrue.shape[0]*yTrue.shape[1]) - 0.95)

def msis(insample, yTrue, yLower, yUpper):
    
    ts_naive_err = naive_error(insample)
    
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
