import pandas as pd
import numpy as np   

from tensorflow.python.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K

class M4DataLoader(object):

    def __init__(self, train_data_path, test_data_path, lookback=48, horizon=48, validation_ratio = 0.10):
        
        self.validation_ratio = validation_ratio
        self.lookback = lookback
        self.horizon = horizon
        
        self.__load_data(train_data_path, test_data_path)
    
    def __read_raw_data(self, file_path):

        df = pd.read_csv(file_path)
        del df['V1']
        return df.values
    
    def __merge_and_standarize(self,raw_train_data, raw_test_data):
        
        # Join train and test to standarize together
        full_data = np.concatenate((raw_train_data, raw_test_data), axis=1)
        
        scaler = StandardScaler()
        scaler.fit(full_data)

        return scaler.transform(full_data)
    
    def __build_augmented_data(self, train_data):
        
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
        
        raw_train_data = self.__read_raw_data(train_data_path)
        raw_test_data = self.__read_raw_data(test_data_path)
        full_data = self.__merge_and_standarize(raw_train_data, raw_test_data)

        validation_data_size = int (full_data.shape[0]*self.validation_ratio)

        self.train_test_data = full_data[:-validation_data_size,:]
        self.validation_data = full_data[-validation_data_size:,:]

        self.train_serie_length = raw_train_data.shape[1]
        self.test_serie_length = raw_test_data.shape[1]


    def get_training_data(self):
        return self.__build_augmented_data(self.train_test_data[:,:self.train_serie_length])
        

    def get_test_data(self):
        return self.__build_from_series_pairs(self.train_test_data[:,:self.train_serie_length],
            self.train_test_data[:,-self.test_serie_length:])

    def get_validation_data(self):
        X1, Y1 = self.__build_augmented_data(self.validation_data[:,:self.train_serie_length])
        X2, Y2 = self.__build_from_series_pairs(self.validation_data[:,:self.train_serie_length],
            self.validation_data[:,-self.test_serie_length:])

        return np.concatenate((X1, X2), axis=0), np.concatenate((Y1, Y2), axis=0)

    