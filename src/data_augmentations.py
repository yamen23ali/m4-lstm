import pandas as pd
import numpy as np   

from src.utils import read_raw_data
from tensorflow.python.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K

class DiffAugmentation(object):
    """ Create new features by taking the difference of each point and the one before it
    """

    def create(self, data):
        """
        Create a new set of features from the original data
        The new feature here is the difference between each point and the one before it.
        
        Example:
            data = (2, 7, 10, 8, 9)
            new_data = (2, 5, 3, 2, 1)

        Args:
            data (array_like): The data to generate new features for

        Returns:
            array_like: The new features to add to the data
        """
        shifted_data = np.hstack((data[:,1:], data[:,-1][:,np.newaxis]))
        return shifted_data - data

class StdAugmentation(object):

    """ Create new features by calculating the STD of a gaussian distribution  for each data point

        Here we divide the data points values into ranges and assign each range a coefficient value.
        The final std value is obtained by multiplying the data point with its corresponding coefficient value.
        So the final std related to each point is a ratio of the point itself.

        Args:
            pi_params (dict): The parameters to use in calculating the std
            epsilon (float): A small value to add for data points with value = 0 in order to avoid any problems during the calculations

        Attributes:
            epsilon (float): A small value to add for data points with value = 0 in order to avoid any problems during the calculations
            max_coff (float): The maximum ratio of a data point to use as std
            min_coff (float): The minimum ratio of a data point to use as std
            step (float): The increment in the coefficient value between 2 consuective ranges
    """

    def __init__(self, pi_params = {}, epsilon = 0.05):
        try:
            self.epsilon = epsilon
            self.max_coff = pi_params['max_coff']
            self.min_coff = pi_params['min_coff']
            self.step = pi_params['step']
        except Exception as e:
            print("Missing PI Params")

    def create(self, data):
        """
        Create a new set of features from the original data
        The new feature here is standard deviation of a gaussian distributions
        given that each data point is considered as a mean of a gaussian disribution.

        The std is computed in a way that insure we don't have too wide or too narrow ranges around the means.
        
        Example:
            data = (0.2, 1.7, 10, 3)
            std =  (0.1, 0.3, 1.4, 0.5)

        Args:
            data (array_like): The data to generate new features for

        Returns:
            array_like: The new features to add to the data
        """
        data = np.abs(data) + self.epsilon

        ranges_number = round((self.max_coff - self.min_coff) / self.step)

        max_values = data.max(axis = 1)
        ranges_step = max_values / ranges_number

        coffs = self.max_coff - (( data / ranges_step[:,np.newaxis]) * self.step)

        return data * coffs

    def set_pi_params(self, pi_params):
        """
        Update the parameters used in calculating the std

        Args:
            pi_params (dict): The parameters to use in calculating the std
        """
        self.max_coff = pi_params['max_coff']
        self.min_coff = pi_params['min_coff']
        self.step = pi_params['step']
