import tensorflow as tf
import numpy as np

def smapetf(yTrue, yPred):
    ratio = tf.abs(yTrue - yPred) / (tf.abs(yTrue) + tf.abs(yPred))
    return tf.reduce_mean(200/48 * tf.reduce_sum(ratio, axis = 1) )

def smape(yTrue, yPred):
    ratio = np.abs(yPred - yTrue) / (np.abs(yTrue) + np.abs(yPred))
    return  np.mean( (200/yTrue.shape[1] ) * np.sum( ratio, axis = 1))