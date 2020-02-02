import pandas as pd
import numpy as np   

from src.utils import read_raw_data

from tensorflow.python.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K

class M4DataLoader(object):

    """ This class is used to load data from csv files and prepare it to be used with the model.
        The data preparation here has multiple aspects :
            - Standarizing the data.
            - Augmenting new features if needed.
            - Split the data into (training, test & holdout)
            - Generate multiple time series form on time serie using the llok back and horizon parameters.


        Args:
            train_path (str): The path of the training.csv file of the training_test dataset
            test_path (str): The path of the test.csv file of the training_test dataset
            train_holdout_path (str): The path of the training.csv file of the holdout dataset
            test_holdout_path (str): The path of the test.csv file of the holdout dataset
            x_augmentations (:obj:`list` of Augmentations): A list of augmentations to apply on the input data (i.e. data fed to the model)
            y_augmentations (:obj:`list` of Augmentations): A list of augmentations to apply on the output data (i.e. predictions of the model)
            lookback (int): How many steps to lookback in the past (i.e. The input of the model)
            horizon (int): The prediction horizon (i.e. how many steps a head to predict)
            

        Attributes:
            x_augmentations (:obj:`list` of Augmentations): A list of augmentations to apply on the input data (i.e. data fed to the model)
            y_augmentations (:obj:`list` of Augmentations): A list of augmentations to apply on the output data (i.e. predictions of the model)
            lookback (int): How many steps to lookback in the past (i.e. The input of the model)
            horizon (int): The prediction horizon (i.e. how many steps a head to predict)
            raw_train_data (array_like): The raw csv train data from the train_test dataset
            raw_test_data (array_like): The raw csv train data from the train_test dataset
            raw_train_holdout_data (array_like): The raw csv train data from the holdout dataset
            raw_test_holdout_data (array_like): The raw csv train data from the holdout dataset
            train_test_data (array_like): The standarized train_test dataset
            holdout_data (array_like): The standarized holdout dataset
            train_serie_length (int): The training time serie length in the train_test dataset
            test_serie_length (int): The test time serie length in the train_test dataset
            train_holdout_serie_length (int): The training time serie length in the holdout dataset
            test_holdout_serie_length (int): The test time serie length in the holdout dataset
    """

    def __init__(self, train_data_path, test_data_path, train_holdout_path, test_holdout_path, 
        x_augmentations =[], y_augmentations = [], lookback=48, horizon=48):
        
        self.lookback = lookback
        self.horizon = horizon
        self.x_augmentations = x_augmentations
        self.y_augmentations = y_augmentations
        
        self.__load_data(train_data_path, test_data_path, train_holdout_path, test_holdout_path)


    def get_training_data(self):
        """ Get the data ( X, Y ) to train the model on

        Returns:
            (array_like, array_like): The data  (X,Y) to train the model on
        """

        X, Y = self.__build_from_series(self.train_test_data[:,:self.train_serie_length])

        return self.__apply_x_augmentations(X), self.__apply_y_augmentations(Y)

    def get_test_data(self):
        """ Get the data ( X, Y ) to test the model on

        Returns:
            (array_like, array_like): The data (X,Y) to test the model on
        """
        X, Y= self.__build_from_series_pairs(self.train_test_data[:,:self.train_serie_length],
            self.train_test_data[:,-self.test_serie_length:])

        return self.__apply_x_augmentations(X), self.__apply_y_augmentations(Y)

    def get_holdout_data(self):
        """ Get the never seen data( X, Y ) to test the model on

        Returns:
            (array_like, array_like): The never seen data (X,Y) to test the model on
        """
        X, Y= self.__build_from_series_pairs(self.holdout_data[:,:self.train_holdout_serie_length],
            self.holdout_data[:,-self.test_holdout_serie_length:])

        return self.__apply_x_augmentations(X), self.__apply_y_augmentations(Y)

    def unstandarize_predictions(self, predictions):
        """ Unstandarize the model predictions

        Returns:
            (array_like): The never seen data to test the model on
        """

        # Put the data together before unstandarizing it
        complete_data = np.append(self.train_test_data, self.holdout_data, axis=0)
        complete_data = complete_data[:,:-48]
        complete_data = np.concatenate((complete_data, predictions), axis=1).T
        
        unstandarize_data = self.scaler.inverse_transform(complete_data).T
        
        return unstandarize_data[:,-48:]
    
    def __merge_and_standarize(self, raw_train_data, raw_test_data, 
        raw_train_holdout_data, raw_test_holdout_data):
        """ Merge all the loaded data together and standarize it
        Here the data comes from 4 differet files (train.csv, test.csv, train_holdout.csv, test_holdout.csv)

         Args:
            raw_train_data (array_like): The loaded train data from training_test dataset
            raw_test_data (array_like): The loaded test data from training_test dataset
            raw_train_holdout_data (array_like): The loaded train data from holdout dataset
            raw_test_holdout_data (array_like): The loaded test data from holdout dataset

        Returns:
            (array_like, array_like): The standarized training and test data from both (train_test, holdout) datasets
        """
        
        # Join train, test and holdout to standarize together
        train_test_data = np.concatenate((raw_train_data, raw_test_data), axis=1)
        holdout_data = np.concatenate((raw_train_holdout_data, raw_test_holdout_data), axis=1)
        complete_data = np.append(train_test_data, holdout_data, axis=0).T

        self.scaler = StandardScaler()
        self.scaler.fit(complete_data)
        standarized_data = self.scaler.transform(complete_data).T

        return standarized_data[:raw_train_data.shape[0], :], standarized_data[raw_train_data.shape[0]:, :]
    
    def __build_from_series(self, data):
        """ Build the training data series from the raw series
        Here we expand our training data by generating multiple time series from  one time serie.
        We generate the series by moving a sliding window of size (lookback) across one serie.

         Args:
            data (array_like): The data to generate the training data from

        Returns:
            (array_like, array_like): The training data of the form (X,Y)
        """
        
        data_x =  np.empty(shape=[0, self.lookback])
        data_y =  np.empty(shape=[0, self.horizon])

        sample_num = 0
        
        while True:
            start_x_indx = sample_num * self.lookback
            end_x_indx = (sample_num+1) * self.lookback
            end_y_indx = end_x_indx + self.horizon
            
            if end_y_indx > data.shape[1]:  break
                
            data_x = np.vstack((data_x, data[:, start_x_indx : end_x_indx ]))
            data_y = np.vstack((data_y, data[:, end_x_indx : end_y_indx]))
            
            sample_num+=1
        

        return data_x[~np.isnan(data_y).any(axis=1)], data_y[~np.isnan(data_y).any(axis=1)]


    def __build_from_series_pairs(self, train_data, test_data):
        """ Build the test and holdout series from the raw series
        Here we take the last (lookback) values from the train_data as X, and the first (horizon) values from test_data as Y.

         Args:
            train_data (array_like): The data we use as input for the model
            test_data (array_like): The data we use as target

        Returns:
            (array_like, array_like): The test(holdout) data of the form (X,Y)
        """
        
        data_x =  np.empty(shape=[0, self.lookback])
        data_y =  test_data[:,:self.horizon]
        
        for ts in train_data:
            ts = ts[~np.isnan(ts)]
            ts = ts[-self.lookback:]
            
            data_x = np.vstack((data_x, ts))
            

        return data_x, data_y
    
    def __load_data(self, train_data_path, test_data_path, train_holdout_path, test_holdout_path):
        """ Load the data from csv files, standarize and split
        Here we take the last (lookback) values from the train_data as X, and the first (horizon) values from test_data as Y.

         Args:
            train_path (str): The path of the training.csv file of the training_test dataset
            test_path (str): The path of the test.csv file of the training_test dataset
            train_holdout_path (str): The path of the training.csv file of the holdout dataset
            test_holdout_path (str): The path of the test.csv file of the holdout dataset
        """

        self.raw_train_data = read_raw_data(train_data_path)
        self.raw_test_data = read_raw_data(test_data_path)
        self.raw_train_holdout_data = read_raw_data(train_holdout_path)
        self.raw_test_holdout_data = read_raw_data(test_holdout_path)

        self.train_test_data, self.holdout_data = self.__merge_and_standarize(self.raw_train_data, self.raw_test_data, 
            self.raw_train_holdout_data, self.raw_test_holdout_data)

        self.train_serie_length = self.raw_train_data.shape[1]
        self.test_serie_length = self.raw_test_data.shape[1]
        self.train_holdout_serie_length = self.raw_train_holdout_data.shape[1]
        self.test_holdout_serie_length = self.raw_test_holdout_data.shape[1]

    def __apply_x_augmentations(self, data):
        """ Apply augmentations on X (i.e. the data used as model input)

         Args:
            data (array_like): The data we want to augment with new features

        Returns:
            (array_like): The augmented data
            
        """
        augmented_data = data
        for augmentation in self.x_augmentations:
            augmented_data = np.dstack((augmented_data, augmentation.create(data) ))
        
        return augmented_data

    def __apply_y_augmentations(self, data):
        """ Apply augmentations on Y (i.e. the data used as model target)

         Args:
            data (array_like): The data we want to augment with new features

        Returns:
            (array_like): The augmented data
            
        """
        augmented_data = data
        for augmentation in self.x_augmentations:
            augmented_data = np.hstack((augmented_data, augmentation.create(data)))
        
        return augmented_data

    

