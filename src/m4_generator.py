import pandas as pd
import numpy as np   

from tensorflow.python.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K

class M4Generator(Sequence):

    def __init__(self, train_data_path, test_data_path, lookback, horizon, batch_size):
        
        self.batch_size = batch_size
        self.lookback = lookback
        self.horizon = horizon
        
        self.__load_data(train_data_path, test_data_path)
    
    def __read_raw_data(self, file_path):
        #float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
        #float32_cols = {c: np.float32 for c in float_cols}
        
        df = pd.read_csv(file_path)
        del df['V1']
        return df.values
    
    def __standarize(self,raw_train_data, raw_test_data):
        
        # Join train and test to standarize together
        full_data = np.concatenate((raw_train_data, raw_test_data), axis=1)
        
        scaler = StandardScaler()
        scaler.fit(full_data)
        full_data = scaler.transform(full_data)
        
        # split to train and test data again
        return full_data[:,:raw_train_data.shape[1]], full_data[:,-raw_test_data.shape[1]:]
    
    def __build_train_data(self, train_data):
        
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

    def __build_test_data(self, train_data, test_data):
        
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
        
        standarized_train_data, standarized_test_data = self.__standarize(raw_train_data, raw_test_data)
        
        self.train_x, self.train_y  = self.__build_train_data(standarized_train_data)
        self.test_x, self.test_y  = self.__build_test_data(standarized_train_data, standarized_test_data)

    def __len__(self):
        return int(np.floor(len(self.train_x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.train_x[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        batch_y = self.train_y[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        

        return np.array(batch_x)[:,:,np.newaxis], np.array(batch_y)[:,:]

    def get_data(self):
        return self.train_x, self.train_y, self.test_x, self.test_y
