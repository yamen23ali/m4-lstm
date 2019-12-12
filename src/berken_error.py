import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K

lambda_ = 24 # lambda in loss fn
alpha_ = 0.05  # capturing (1-alpha)% of samples
soften_ = 150.
r_ = 10^5
decay_lambda = 1/48

def weighted_MASE(y_true, y_pred, horizon):
    
    n = tf.shape(y_true)[0]
    e = tf.abs(tf.subtract(y_true, y_pred))
    max_e = tf.reduce_max(e)
    s_diff = tf.reduce_sum(tf.abs(tf.subtract(y_true[1:],y_true[:-1])), 0)
    d = 1/tf.multiply(tf.cast(n-1, tf.float32),s_diff)
    
    q = e/tf.expand_dims(d, 0)
    
    w = tf.tile(tf.range(horizon), [tf.cast(n/horizon, tf.int32)])
    w_q = q * tf.exp(-decay_lambda * tf.cast(w, tf.float32))
    
    MASE = tf.reduce_mean(w_q, 0)
    
    return MASE

def wMASE_loss(y_true, y_pred):
    
    horizon = tf.shape(y_true)[1]
    y_t = tf.reshape(y_true, [-1])
    y_f = tf.reshape(y_pred[:,1::3], [-1])
    
    n = tf.shape(y_t)[0]
    e = tf.abs(tf.subtract(y_t, y_f))
    max_e = tf.reduce_max(e)
    s_diff = tf.reduce_sum(tf.abs(tf.subtract(y_t[1:],y_t[:-1])), 0)
    d = 1/tf.multiply(tf.cast(n-1, tf.float32),s_diff)
    
    q = e/tf.expand_dims(d, 0)
    
    w = tf.tile(tf.range(horizon), [tf.cast(n/horizon, tf.int32)])
    w_q = q * tf.exp(-decay_lambda * tf.cast(w, tf.float32))
    
    MASE = tf.reduce_mean(w_q, 0)
    
    return MASE

def MASE_loss(y_true, y_pred):
    horizon = tf.shape(y_true)[1]
    y_t = tf.reshape(y_true, [-1])
    y_f = tf.reshape(y_pred[:,1::3], [-1])
    
    n = tf.shape(y_t)[0]
    e = tf.abs(tf.subtract(y_t, y_f))
    max_e = tf.reduce_max(e)
    s_diff = tf.reduce_sum(tf.abs(tf.subtract(y_t[1:],y_t[:-1])), 0)
    d = 1/tf.multiply(tf.cast(n-1, tf.float32),s_diff)
    
    q = e/tf.expand_dims(d, 0)
    
    return tf.reduce_mean(q, 0)

def PICP_loss(y_true, y_pred):
    y_t = tf.reshape(y_true, [-1])
    y_u = tf.reshape(y_pred[:,::3], [-1])
    y_l = tf.reshape(y_pred[:,2::3], [-1])
    
    K_HU = tf.maximum(0.,tf.sign(y_u - y_t))
    K_HL = tf.maximum(0.,tf.sign(y_t - y_l))
    K_H = tf.multiply(K_HU, K_HL)

    return tf.reduce_mean(K_H)



def MPIW_b_loss(y_true, y_pred):

    y_t = tf.reshape(y_true, [-1])
    y_u = tf.reshape(y_pred[:,::3], [-1])
    y_l = tf.reshape(y_pred[:,2::3], [-1])
    
    K_HU = tf.maximum(0.,tf.sign(y_u - y_t))
    K_HL = tf.maximum(0.,tf.sign(y_t - y_l))
    K_H = tf.multiply(K_HU, K_HL)
    
    C = tf.reduce_sum(K_H)
    
    return tf.reduce_mean(tf.multiply((y_u - y_l),K_H)) * C/tf.cast(tf.shape(K_H), tf.float32) + tf.reduce_mean(tf.abs(K_H-1))**(C)

def qd_objective_lstm_b(y_true, y_pred):
    '''Loss_QD-soft, from algorithm 1'''
    h = tf.shape(y_true)[1]
    y_t = tf.reshape(y_true, [-1])
    y_u = tf.reshape(y_pred[:,::3], [-1])
    y_f = tf.reshape(y_pred[:,1::3], [-1])
    y_l = tf.reshape(y_pred[:,2::3], [-1])
    
    wMASE = weighted_MASE(y_t, y_f, h)
    
    K_HU = tf.maximum(0.,tf.sign(y_u - y_t))
    K_HL = tf.maximum(0.,tf.sign(y_t - y_l))
    K_H = tf.multiply(K_HU, K_HL)
    
    K_SU = tf.sigmoid(soften_ * (y_u - y_t))
    K_SL = tf.sigmoid(soften_ * (y_t - y_l))
    K_S = tf.multiply(K_SU, K_SL)
    
    C = tf.reduce_sum(K_H)
    MPIW_b = tf.reduce_mean(tf.multiply((y_u - y_l),K_H)) * C/tf.cast(tf.shape(K_H), tf.float32) + tf.reduce_mean(tf.abs(K_H-1))**(C)
    PICP_H = tf.reduce_mean(K_H)
    PICP_S = tf.reduce_mean(K_S)
    
    Loss_S = MPIW_b + lambda_ * (n_ / (alpha_*(1-alpha_)) * tf.maximum(0.,(1-alpha_) - PICP_S) + wMASE)
    return Loss_S


def wMASE(yTure, yPred):
	timesteps = tf.range(10)
    multiples = tf.constant([yTrue.shape[0]])
    weights = tf.reshape(tf.tile(timesteps, multiples), [ multiples[0], tf.shape(timesteps)[0]])
    weights = tf.exp((-1.0/48.0) * weights)


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
    
    h = tf.shape(y_true)[1]
    y_t = tf.reshape(y_true, [-1])
    y_u = tf.reshape(y_pred[:,::3], [-1])
    y_f = tf.reshape(y_pred[:,1::3], [-1])
    y_l = tf.reshape(y_pred[:,2::3], [-1])
    
    wMASE = weighted_MASE(y_t, y_f, h)
    
    K_HU = tf.maximum(0.,tf.sign(y_u - y_t))
    K_HL = tf.maximum(0.,tf.sign(y_t - y_l))
    K_H = tf.multiply(K_HU, K_HL)
    
    K_SU = tf.sigmoid(soften_ * (y_u - y_t))
    K_SL = tf.sigmoid(soften_ * (y_t - y_l))
    K_S = tf.multiply(K_SU, K_SL)
    
    MPIW_c = tf.reduce_sum(tf.multiply((y_u - y_l),K_H))/(tf.reduce_sum(K_H) + 1)
    PICP_H = tf.reduce_mean(K_H)
    PICP_S = tf.reduce_mean(K_S)
    
    Loss_S = MPIW_c + lambda_ * (n_ / (alpha_*(1-alpha_)) * tf.maximum(0.,(1-alpha_) - PICP_S) + wMASE)
    return Loss_S

def loss_qd(yTrue, yPred):
	yPoint = yPred[:,:48]
	yUpper = yPred[:,48:-48]
	yLower = yPred[:,-48:]














