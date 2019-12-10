import pandas as pd
import numpy as np   

from src.utils import read_raw_data

from tensorflow.python.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K

class M4DataLoader(object):

    def __init__(self, train_data_path, test_data_path, lookback=48, horizon=48, validation_ratio = 0.05, pi_params = {}):
        
        self.validation_ratio = validation_ratio
        self.lookback = lookback
        self.horizon = horizon
        self.pi_params = pi_params
        
        self.__load_data(train_data_path, test_data_path)
    
    
    def __merge_and_standarize(self,raw_train_data, raw_test_data):
        
        # Join train and test to standarize together
        complete_data = np.concatenate((raw_train_data, raw_test_data), axis=1).T
        
        scaler = StandardScaler()
        scaler.fit(complete_data)

        return scaler.transform(complete_data).T
    
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
        
        raw_train_data = read_raw_data(train_data_path)
        raw_test_data = read_raw_data(test_data_path)
        complete_data = self.__merge_and_standarize(raw_train_data, raw_test_data)

        validation_data_size = int (complete_data.shape[0]*self.validation_ratio)

        self.train_test_data = complete_data[:-validation_data_size,:]
        self.validation_data = complete_data[-validation_data_size:,:]

        self.train_serie_length = raw_train_data.shape[1]
        self.test_serie_length = raw_test_data.shape[1]


    def __get_diff(self, data):
        shifted_data = np.hstack((data[:,1:], data[:,-1][:,np.newaxis]))
        return shifted_data - data

    def __get_std(self, data):
        data = np.abs(data) + 0.2

        max_coff = self.pi_params['max_coff']
        min_coff = self.pi_params['min_coff']
        step = self.pi_params['step']

        ranges_number = round((max_coff - min_coff) / step)

        #print(ranges_number)

        max_values = data.max(axis = 1)
        ranges_step = max_values / ranges_number

        coffs = max_coff - (( data / ranges_step[:,np.newaxis]) * step)
        
        #print("==== data")
        #print(data[0,:])
        #print("==== coffs")
        #print(coffs[0,:])

        #print("==== std")
        #print((data * coffs)[0,:])

        return data * coffs



    def __augment_diff_x(self, data):
        #diff = self.__get_diff(data)
        diff = self.__get_std(data)
        return np.dstack((data, diff))

    def __augment_diff_y(self, data):
        #diff = self.__get_diff(data)
        diff = self.__get_std(data)
        return np.hstack((data, diff))

    def get_training_data(self):
        X, Y = self.__build_from_series(self.train_test_data[:,:self.train_serie_length])

        return self.__augment_diff_x(X), self.__augment_diff_y(Y)
        #return X, Y
        

    def get_test_data(self):
        X, Y= self.__build_from_series_pairs(self.train_test_data[:,:self.train_serie_length],
            self.train_test_data[:,-self.test_serie_length:])

        return self.__augment_diff_x(X), self.__augment_diff_y(Y)
        #return X, Y

    def get_validation_data(self):
        X1, Y1 = self.__build_from_series(self.validation_data[:,:self.train_serie_length])
        X2, Y2 = self.__build_from_series_pairs(self.validation_data[:,:self.train_serie_length],
            self.validation_data[:,-self.test_serie_length:])

        X = np.concatenate((X1, X2), axis=0)
        Y = np.concatenate((Y1, Y2), axis=0)

        return self.__augment_diff_x(X), self.__augment_diff_y(Y)
        #return X, Y

