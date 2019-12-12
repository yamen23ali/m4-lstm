import pandas as pd
import numpy as np   

from src.utils import read_raw_data

from tensorflow.python.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K

class DiffAugmentation(object):

    def create(self, data):
        shifted_data = np.hstack((data[:,1:], data[:,-1][:,np.newaxis]))
        return shifted_data - data

class StdAugmentation(object):

    def __init__(self, pi_params, epsilon = 0.05):
        self.max_coff = pi_params['max_coff']
        self.min_coff = pi_params['min_coff']
        self.step = pi_params['step']
        self.epsilon = epsilon

    def create(self, data):
        data = np.abs(data) + self.epsilon

        ranges_number = round((self.max_coff - self.min_coff) / self.step)

        max_values = data.max(axis = 1)
        ranges_step = max_values / ranges_number

        coffs = self.max_coff - (( data / ranges_step[:,np.newaxis]) * self.step)

        return data * coffs