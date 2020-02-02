import tensorflow as tf
import numpy as np

from src.training_loss_functions import *
from tensorflow.keras import backend as K

class BerkenLoss(object):

    """ This class contains an implementation for the loss function suggested in Berken's thesis

        Args:
            horizon (int): The prediction horizon (i.e. how many steps a head to predict)
            batch_size (int): The training batch size
            weighted (bool): Should the loss function use the weighted MASE or not
            lambda_ (int): A lever value to compensate weighten the PICP value
            alpha (float): The confidence level of values inbetween the estimated lower and upper bounds
            soften: (float): A softening factor

        Attributes:
            horizon (int): The prediction horizon (i.e. how many steps a head to predict)
            batch_size (int): The training batch size
            weighted (bool): Should the loss function use the weighted MASE or not
            lambda_ (int): A lever value to compensate weighten the PICP value
            alpha (float): The confidence level of values inbetween the estimated lower and upper bounds
            soften: (float): A softening factor
            mase_func (func): The MASE function to use in the loss function, either weighted or not based on the (weighted) argument value
    """

    def __init__(self, horizon, batch_size, weighted=False, lambda_=24, alpha=0.05, soften=150):
        self.horizon = horizon
        self.batch_size = batch_size
        self.lambda_=lambda_
        self.alpha=alpha
        self.soften = soften
        
        self.mase_func = mase
        
        if weighted: self.mase_func = self.__weighted_mase__

    def loss(self):
        """
        A wrapper for the actual loss function in order to be able to use it in the training
        """
        def wrapper(yTrue, yPred):
            return self.__qd_objective_lstm_c__(yTrue, yPred)
        return wrapper

    def __qd_objective_lstm_c__(self, yTrue, yPred):

        """
        An implementation of the loss function suggested by Berken in his thesis.

        Args:
            yTrue (array_like): The actual timeseries points of the horizon
            yPred (array_like): The predicted timeseries points, upper bounds and lower bounds of the horizon

        Returns:
            array_like: The value of the calculated loss per sample

        """

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
        """
        An implementation of the weighted MASE suggested by Berken in his thesis.

        Args:
            yTrue (array_like): The actual timeseries points of the horizon
            yPred (array_like): The predicted timeseries points, upper bounds and lower bounds of the horizon

        Returns:
            array_like: The value of the calculated weighted MASE per sample

        """
        n = tf.shape(yTrue)[0]
        abs_err = tf.abs(tf.subtract(yTrue, yPred))
        naive_err = tf.reduce_mean(tf.abs(yTrue[:,1:] - yTrue[:,:-1]), axis=1)
        
        q = abs_err/tf.expand_dims(naive_err, axis=1)
        
        w = tf.tile(tf.range(self.horizon), [self.batch_size])
        w = tf.reshape(w, [self.batch_size, self.horizon])
        
        #weight the errors
        w_q = q * tf.exp(-self.lambda_ * tf.cast(w, tf.float64))

        return tf.reduce_mean(w_q, axis=1)

