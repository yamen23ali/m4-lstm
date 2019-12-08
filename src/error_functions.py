import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K

def kl_div(mean_true, mean_pred, std_true, std_pred):
	d = std_true**2 + (mean_true - mean_pred)**2
	a = 2 * std_pred**2
	kl = K.log(std_pred / std_true) + (d/a) - 0.5

	return tf.reduce_mean(kl, axis = 1)[:,np.newaxis]

def yamen(yTrue, yPred):
	tsTrue = yTrue[:,:48]
	tsPred = yPred[:,:48]

	stdTrue =  0.2 # same for all yTrue points
	
	stdPred = tf.abs(yPred[:,-48:])

	bounds_err = kl_div(tsTrue, tsPred, stdTrue, stdPred)

	return  bounds_err #tf.add(bounds_err, smape(tsTrue, tsPred))

def naive_error(yTrue):
	return tf.reduce_mean(tf.abs(yTrue[:,1:] - yTrue[:,:-1]), axis=1)

def mase(yTrue, yPred):
	naive_err = naive_error(yTrue)
	return mae(yTrue, yPred)[:,np.newaxis] / naive_err[:,np.newaxis]

def mae(yTrue, yPred):
	return tf.reduce_mean(tf.abs(yTrue - yPred), axis=1)

def rmse(yTrue, yPred):
	return tf.sqrt( tf.reduce_mean(tf.square(yTrue - yPred), axis=1))

def smape(yTrue, yPred):
    ratio = tf.abs(yTrue - yPred) / (tf.abs(yTrue) + tf.abs(yPred))
    return 200/yPred.shape[1] * tf.reduce_sum(ratio, axis = 1)