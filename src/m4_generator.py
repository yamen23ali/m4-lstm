import pandas as pd
import numpy as np   

from tensorflow.python.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K

class M4Generator(Sequence):

    """ 
        This class implements a data generator that will return data in batches

        Args:
            X (array_like): The insample data ( i.e. The timeseries fed to the model as input)
            Y (array_like): The target data (i.e. The expected timesereis at the output)
            batch_size (int): The number of samples in one batch
            features_number (int): The number of features in input data

        Attributes:
            X (array_like): The insample data ( i.e. The timeseries fed to the model as input)
            Y (array_like): The target data (i.e. The expected timesereis at the output)
            batch_size (int): The number of samples in one batch
            features_number (int): The number of features in input data
    """

    def __init__(self, X, Y, batch_size, features_number):
        
        self.batch_size = batch_size
        self.X = X
        self.Y = Y
        self.features_number = features_number

    def __len__(self):
        """
            Get the total number of batches

            Returns:
                int: The total number of batches
        """
        return int(np.floor(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
            Get a pair of items (input, target) based on their index in the data array

            Args:
                idx (int): The item index in the data array

            Returns:
                (array_like, array_like): (input timeserie, target timeserie)
        """
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        batch_y = self.Y[idx * self.batch_size:(idx + 1) * self.batch_size, :]

        batch_x =  batch_x.reshape(batch_x.shape[0],batch_x.shape[1], self.features_number)
        
        return batch_x, batch_y

    def steps_per_epoch(self):
        """
            Get the number of steps in each epochs (i.e. the total number of batches)

            Returns:
                int: The numebr of steps in each epochs
        """
        return self.__len__()
