import tensorflow as tf
import numpy as np

from src.training_loss_functions import *
from tensorflow.keras import backend as K

lambda_ = 24 # lambda in loss fn
alpha_ = 0.05  # capturing (1-alpha)% of samples
soften_ = 150.



def PICP_loss(yTrue, upper, lower):
    y_t = tf.reshape(yTrue, [-1])
    y_u = tf.reshape(upper, [-1])
    y_l = tf.reshape(lower, [-1])
    
    K_HU = tf.maximum(0.,tf.sign(y_u - y_t))
    K_HL = tf.maximum(0.,tf.sign(y_t - y_l))
    K_H = tf.multiply(K_HU, K_HL)

    return tf.reduce_mean(K_H)

def MPIW_c_loss(yTrue, upper, lower):
    y_t = tf.reshape(yTrue, [-1])
    y_u = tf.reshape(upper, [-1])
    y_l = tf.reshape(lower, [-1])
    
    K_HU = tf.maximum(0.,tf.sign(y_u - y_t))
    K_HL = tf.maximum(0.,tf.sign(y_t - y_l))
    K_H = tf.multiply(K_HU, K_HL)
    
    return tf.reduce_sum(tf.multiply((y_u - y_l),K_H))/(tf.reduce_sum(K_H) + 1)

def qd_objective_lstm_c(yTrue, yPred):
    '''Loss_QD-soft, from algorithm 1'''
    
    h = 48
    batch_size = 120
    zero = tf.cast(0.,tf.float64)
    y_t = tf.reshape(yTrue, [-1])
    y_u = tf.reshape(yPred[:,48:-48], [-1])
    y_f = tf.reshape(yPred[:,:48], [-1])
    y_l = tf.reshape(yPred[:,-48:], [-1])
    
    wMASE = mase(yTrue, yPred[:,:48])
    
    K_HU = tf.maximum(zero, tf.sign(y_u - y_t))
    K_HL = tf.maximum(zero, tf.sign(y_t - y_l))
    K_H = tf.multiply(K_HU, K_HL)
    
    K_SU = tf.sigmoid(soften_ * (y_u - y_t))
    K_SL = tf.sigmoid(soften_ * (y_t - y_l))
    K_S = tf.multiply(K_SU, K_SL)
    
    MPIW_c = tf.reduce_sum(tf.multiply((y_u - y_l),K_H))/(tf.reduce_sum(K_H) + 1)
    PICP_H = tf.reduce_mean(K_H)
    PICP_S = tf.reduce_mean(K_S)
    
    Loss_S = MPIW_c + lambda_ * (batch_size/ (alpha_*(1-alpha_)) * tf.maximum(zero, (1-alpha_) - PICP_S) + wMASE)
    return Loss_S
