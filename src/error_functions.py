import tensorflow as tf
import numpy as np

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