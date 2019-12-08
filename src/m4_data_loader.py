import pandas as pd
import numpy as np   

from src.utils import read_raw_data

from tensorflow.python.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K

class M4DataLoader(object):

    def __init__(self, train_data_path, test_data_path, lookback=48, horizon=48, validation_ratio = 0.05):
        
        self.validation_ratio = validation_ratio
        self.lookback = lookback
        self.horizon = horizon
        
        self.__load_data(train_data_path, test_data_path)

        
    def get_training_data(self):
        X, Y = self.__build_from_series(self.train_test_data[:,:self.train_serie_length])
        X_extended_dim = self.__build_lower_upper_dimensions_x(self.__augment_diff_x(X))
        Y_extended_dim = self.__build_lower_upper_dimensions_y(self.__augment_diff_y(Y),self.horizon)
        
        return X_extended_dim, Y_extended_dim
        

    def get_test_data(self):
        X, Y= self.__build_from_series_pairs(self.train_test_data[:,:self.train_serie_length],
            self.train_test_data[:,-self.test_serie_length:])

        X_extended_dim = self.__build_lower_upper_dimensions_x(self.__augment_diff_x(X))
        Y_extended_dim = self.__build_lower_upper_dimensions_y(self.__augment_diff_y(Y),self.horizon)
        
        return X_extended_dim, Y_extended_dim

    
    def get_validation_data(self):
        X1, Y1 = self.__build_from_series(self.validation_data[:,:self.train_serie_length])
        X2, Y2 = self.__build_from_series_pairs(self.validation_data[:,:self.train_serie_length],
            self.validation_data[:,-self.test_serie_length:])

        X = np.concatenate((X1, X2), axis=0)
        Y = np.concatenate((Y1, Y2), axis=0)

        X_extended_dim = self.__build_lower_upper_dimensions_x(self.__augment_diff_x(X))
        Y_extended_dim = self.__build_lower_upper_dimensions_y(self.__augment_diff_y(Y),self.horizon)
        
        return X_extended_dim, Y_extended_dim

    
    def __load_data(self, train_data_path, test_data_path):
        
        raw_train_data = read_raw_data(train_data_path)
        raw_test_data = read_raw_data(test_data_path)
        complete_data = self.__merge_and_standarize(raw_train_data, raw_test_data)

        validation_data_size = int (complete_data.shape[0]*self.validation_ratio)

        self.train_test_data = complete_data[:-validation_data_size,:]
        self.validation_data = complete_data[-validation_data_size:,:]

        self.train_serie_length = raw_train_data.shape[1]
        self.test_serie_length = raw_test_data.shape[1]


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
 
    

    def __build_lower_upper_dimensions_x(self, data, sigma=0.1):    
        enhanced_data = []
        for timeseries in data:
            enhanced_data.append(self.__get_enhanced_timeseries_lower_upper_x(timeseries, sigma=0.1))

        return np.array(enhanced_data)


    def __build_lower_upper_dimensions_y(self, data, horizon, sigma=0.1):    
        enhanced_data = []
        for timeseries in data:
            enhanced_data.append(self.__get_enhanced_timeseries_lower_upper_y(timeseries,horizon, sigma=0.1))
        
        return np.array(enhanced_data)


    def __get_enhanced_timeseries_lower_upper_x(self, timeseries, sigma=0.1):
        enhanced_timeseries = []

        for timestep, diff in timeseries:
            lower_bound, upper_bound = self.__get_lower_upper_bound(timestep, sigma=0.1)
            enhanced_timeseries.append([timestep, diff, lower_bound, upper_bound])

        return np.array(enhanced_timeseries)
   
    def __get_enhanced_timeseries_lower_upper_y(self, timeseries, horizon, sigma=0.1):
        lower_bounds = []
        upper_bounds = []
        for i in range(0, horizon):
            lower_bound, upper_bound = self.__get_lower_upper_bound(timeseries[i], sigma=0.1)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)

        return np.hstack((timeseries, lower_bounds, upper_bounds))


    def __get_lower_upper_bound(self, mu, sigma=0.1):
        lower_bound = sigma * np.random.uniform(0.1,2) - mu
        upper_bound = sigma * np.random.uniform(0.1,2) + mu

        return lower_bound, upper_bound


    def __get_diff(self, data):
        shifted_data = np.hstack((data[:,1:], data[:,-1][:,np.newaxis]))
        return shifted_data - data

    def __augment_diff_x(self, data):
        diff = self.__get_diff(data)
        return np.dstack((data, diff))

    def __augment_diff_y(self, data):
        diff = self.__get_diff(data)
        return np.hstack((data, diff))
