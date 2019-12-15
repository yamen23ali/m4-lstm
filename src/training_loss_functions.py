import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K

def __kl_div_gaussian(mean_true, mean_pred, std_true, std_pred):
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

def kl_divergance_diff(yTrue, yPred):
	tsTrue = yTrue[:,:48]
	tsPred = yPred[:,:48]

	stdTrue = yTrue[:,-48:] 
	stdPred = tf.abs(yPred[:,-48:])
	err = __kl_div_gaussian(tsTrue, tsTrue, stdTrue, stdPred)

	return  tf.add(err, mase(yTrue[:,:96], yPred[:,:96]))

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
