import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K

def __kl_div_gaussian(mean_true, mean_pred, std_true, std_pred):
	"""
		Implementation of KL divergance between 2 vectors of gaussians

		Args:
			mean_true (array_like): The target means (i.e. the gaussians means of the first gaussians vector)
			mean_pred (array_like): The predicted means (i.e. the gaussians means of the second gaussians vector)
			std_true (array_like): The target stds (i.e. the gaussians stds of the first gaussians vector)
			std_pred (array_like): The predicted stds (i.e. the gaussians stds of the second gaussians vector)

		Returns:
			(list): A list of kl divergances
    """
	d = std_true**2 + (mean_true - mean_pred)**2
	a = 2 * std_pred**2
	kl = K.log(std_pred / std_true) + (d/a) - 0.5

	return tf.reduce_mean(kl, axis = 1)[:,np.newaxis]


def kl_divergance(yTrue, yPred):
	"""
		Calculate the KL divergance between targets and predictions

		Args:
			yTrue (array_like): The targets (i.e. the target gaussians means and stds)
			yPred (array_like): The predictions (i.e. the predicted gaussians means and stds)

		Returns:
			(float): The kl divergance error
    """
	tsTrue = yTrue[:,:48]
	tsPred = yPred[:,:48]

	stdTrue = yTrue[:,48:] 
	stdPred = tf.abs(yPred[:,-48:])

	return  __kl_div_gaussian(tsTrue, tsPred, stdTrue, stdPred)

def naive_error(yTrue):
	"""
		Calculate the error obtained by a naive model

		Args:
			yTrue (array_like): The target points

		Returns:
			(float): The naive model error
    """
	return tf.reduce_mean(tf.abs(yTrue[:,1:] - yTrue[:,:-1]), axis=1)

def mase(yTrue, yPred):
	"""
		Calculate the MASE ( Mean Absolute Scaled Error)

		Args:
			yTrue (array_like): The target points
			yPred (array_like): The predicted points

		Returns:
			(float): The MASE error
    """
	naive_err = naive_error(yTrue)
	return mae(yTrue, yPred)[:,np.newaxis] / naive_err[:,np.newaxis]
	
def mae(yTrue, yPred):
	"""
		Calculate the MAE (Mean Absolute Error) error

		Args:
			yTrue (array_like): The target points
			yPred (array_like): The predicted points

		Returns:
			(float): The MAE error
    """
	return tf.reduce_mean(tf.abs(yTrue - yPred), axis=1)

def rmse(yTrue, yPred):
	"""
		Calculate the RMSE ( Root Mean Squared Error)

		Args:
			yTrue (array_like): The target points
			yPred (array_like): The predicted points

		Returns:
			(float): The RMSE error
    """
	return tf.sqrt( tf.reduce_mean(tf.square(yTrue - yPred), axis=1))

def smape(yTrue, yPred):
	"""
		Calculate the SMAPE ( Symmetric Mean Absolute Percentage Error)

		Args:
			yTrue (array_like): The target points
			yPred (array_like): The predicted points

		Returns:
			(float): The SMAPE error
    """
    ratio = tf.abs(yTrue - yPred) / (tf.abs(yTrue) + tf.abs(yPred))
    return 200/yPred.shape[1] * tf.reduce_sum(ratio, axis = 1)
