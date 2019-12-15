import pandas as pd
import numpy as np   

from tensorflow.python.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K

class M4Generator(Sequence):

    def __init__(self, X, Y, batch_size, features_number):
        
        self.batch_size = batch_size
        self.X = X
        self.Y = Y
        self.features_number = features_number

    def __len__(self):
        return int(np.floor(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        batch_y = self.Y[idx * self.batch_size:(idx + 1) * self.batch_size, :]

        batch_x =  batch_x.reshape(batch_x.shape[0],batch_x.shape[1], self.features_number)
        
        return batch_x, batch_y

    def steps_per_epoch(self):
        return self.__len__()
