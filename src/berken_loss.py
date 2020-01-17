import tensorflow as tf
import numpy as np

from src.training_loss_functions import *
from tensorflow.keras import backend as K

class BerkenLoss(object):

    def __init__(self, horizon, batch_size, weighted=False, lambda_=24, alpha=0.05, soften=150):
        self.horizon = horizon
        self.batch_size = batch_size
        self.lambda_=lambda_
        self.alpha=alpha
        self.soften = soften
        
        self.mase_func = mase
        
        if weighted: self.mase_func = self.__weighted_mase__

    def loss(self):
        def wrapper(yTrue, yPred):
            return self.__qd_objective_lstm_c__(yTrue, yPred)
        return wrapper

    def __qd_objective_lstm_c__(self, yTrue, yPred):

        MASE = self.mase_func(yTrue, yPred[:,:self.horizon])

        zero = tf.cast(0.,tf.float64)
        y_t = tf.reshape(yTrue, [-1])
        y_u = tf.reshape(yPred[:,self.horizon:-self.horizon], [-1])
        y_f = tf.reshape(yPred[:,:self.horizon], [-1])
        y_l = tf.reshape(yPred[:,-self.horizon:], [-1])

        K_HU = tf.maximum(zero, tf.sign(y_u - y_t))
        K_HL = tf.maximum(zero, tf.sign(y_t - y_l))
        K_H = tf.multiply(K_HU, K_HL)

        K_SU = tf.sigmoid(self.soften * (y_u - y_t))
        K_SL = tf.sigmoid(self.soften * (y_t - y_l))
        K_S = tf.multiply(K_SU, K_SL)

        MPIW_c = tf.reduce_sum(tf.multiply((y_u - y_l),K_H))/(tf.reduce_sum(K_H) + 1)
        PICP_H = tf.reduce_mean(K_H)
        PICP_S = tf.reduce_mean(K_S)

        Loss_S = MPIW_c + self.lambda_ * (self.batch_size/ (self.alpha*(1-self.alpha)) * tf.maximum(zero, (1-self.alpha) - PICP_S) + MASE)

        return Loss_S

    def __weighted_mase__(self, yTrue, yPred):
        n = tf.shape(yTrue)[0]
        abs_err = tf.abs(tf.subtract(yTrue, yPred))
        naive_err = tf.reduce_mean(tf.abs(yTrue[:,1:] - yTrue[:,:-1]), axis=1)
        
        q = abs_err/tf.expand_dims(naive_err, axis=1)
        
        w = tf.tile(tf.range(self.horizon), [self.batch_size])
        w = tf.reshape(w, [self.batch_size, self.horizon])
        
        #weight the errors
        w_q = q * tf.exp(-self.lambda_ * tf.cast(w, tf.float64))

        return tf.reduce_mean(w_q, axis=1)

