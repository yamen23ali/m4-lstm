import pandas as pd
import numpy as np   

from src.utils import read_raw_data

from tensorflow.python.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K

class M4DataLoader(object):

    def __init__(self, train_data_path, test_data_path, x_augmentations =[], y_augmentations = [], lookback=48, horizon=48):
        
        self.lookback = lookback
        self.horizon = horizon
        self.x_augmentations = x_augmentations
        self.y_augmentations = y_augmentations
        
        self.__load_data(train_data_path, test_data_path)


    def get_training_data(self):
        X, Y = self.__build_from_series(self.train_test_data[:,:self.train_serie_length])

        return self.__apply_x_augmentations(X), self.__apply_y_augmentations(Y)

    def get_test_data(self):
        X, Y= self.__build_from_series_pairs(self.train_test_data[:,:self.train_serie_length],
            self.train_test_data[:,-self.test_serie_length:])

        return self.__apply_x_augmentations(X), self.__apply_y_augmentations(Y)

    def unstandarize_predictions(self, predictions):


        data = self.train_test_data[:,:-48]

        complete_data = np.concatenate((data, predictions), axis=1).T
        unstandarize_data = self.scaler.inverse_transform(complete_data).T
        
        return unstandarize_data[:,-48:]
    
    def __merge_and_standarize(self,raw_train_data, raw_test_data):
        
        # Join train and test to standarize together
        complete_data = np.concatenate((raw_train_data, raw_test_data), axis=1).T
        
        self.scaler = StandardScaler()
        self.scaler.fit(complete_data)

        return self.scaler.transform(complete_data).T
    
    def __build_from_series(self, train_data):
        
        data_x =  np.empty(shape=[0, self.lookback])
        data_y =  np.empty(shape=[0, self.horizon])

        sample_num = 0
        
        while True:
            start_x_indx = sample_num * self.lookback
            end_x_indx = (sample_num+1) * self.lookback
            end_y_indx = end_x_indx + self.horizon
            
            if end_y_indx > train_data.shape[1]:  break
                
            data_x = np.vstack((data_x, train_data[:, start_x_indx : end_x_indx ]))
            data_y = np.vstack((data_y, train_data[:, end_x_indx : end_y_indx]))
            
            sample_num+=1
        

        return data_x[~np.isnan(data_y).any(axis=1)], data_y[~np.isnan(data_y).any(axis=1)]


    def __build_from_series_pairs(self, train_data, test_data):
        
        data_x =  np.empty(shape=[0, self.lookback])
        data_y =  test_data[:,:self.horizon]
        
        for ts in train_data:
            ts = ts[~np.isnan(ts)]
            ts = ts[-self.lookback:]
            
            data_x = np.vstack((data_x, ts))
            

        return data_x, data_y
    
    def __load_data(self, train_data_path, test_data_path):

        self.raw_train_data = read_raw_data(train_data_path)
        self.raw_test_data = read_raw_data(test_data_path)

        self.train_test_data = self.__merge_and_standarize(self.raw_train_data, self.raw_test_data)

        self.train_serie_length = self.raw_train_data.shape[1]
        self.test_serie_length = self.raw_test_data.shape[1]

    def __apply_x_augmentations(self, data):
        augmented_data = data
        for augmentation in self.x_augmentations:
            augmented_data = np.dstack((augmented_data, augmentation.create(data) ))
        
        return augmented_data

    def __apply_y_augmentations(self, data):
        augmented_data = data
        for augmentation in self.x_augmentations:
            augmented_data = np.hstack((augmented_data, augmentation.create(data)))
        
        return augmented_data

    

