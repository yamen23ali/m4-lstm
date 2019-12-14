import keras
import tensorflow as tf
import json
import numpy as np

from src.utils import create_model_dir

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
from keras.models import model_from_json


class M4Model(object):

    def __init__(self, hidden_layer_size=100, batch_size=50, lookback=48, 
        output_size=48, learning_rate=0.001, loss='mae', dropout_ratio=0.0, features_number = 1, pi_params={}, callbacks=[]):

        self.architecture_file_name = 'architecture.json'
        self.weights_file_name = 'weights.h5'
        self.hyperparameters_file_name = 'hyperparameters.json'

        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.lookback = lookback
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.dropout_ratio = dropout_ratio
        self.features_number = features_number
        self.pi_params = pi_params
        self.callbacks = callbacks

        self.model = Sequential()


        #self.model.add(LSTM(hidden_layer_size, batch_input_shape=(batch_size, lookback, features_number), return_sequences=True, activation='tanh',
        #   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1), recurrent_dropout=dropout_ratio))

        #self.model.add(LSTM(hidden_layer_size, batch_input_shape=(batch_size, lookback,features_number), return_sequences=True, activation='tanh',
        #   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.3), recurrent_dropout=dropout_ratio))

        self.model.add(LSTM(hidden_layer_size, batch_input_shape=(batch_size, lookback,features_number),  activation='tanh',
              kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.5), recurrent_dropout=dropout_ratio))

        self.model.add(Dense(output_size, activation='linear',
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)))

        self.opt = optimizers.RMSprop(lr=learning_rate)#, decay=0.001)#, clipvalue=0.5)#, decay=0.5/20.0, clipvalue=0.3) #decay=0.1/20.0, ) #, clipvalue=1.5) #, decay=1e-3, clipvalue=0.1) #  clipvalue=0.1) #
        #self.opt = optimizers.SGD(lr=learning_rate, decay=1e-2, momentum=0.7, nesterov=True)

        self.model.compile(loss=self.loss, optimizer=self.opt)

    def compile(self):
        self.model.compile(loss=self.loss, optimizer=self.opt)

    def train(self, training_data_generator, test_data_generator, epochs):
        self.epochs = epochs

        return self.model.fit_generator(training_data_generator,
            validation_data = test_data_generator,
            validation_steps=test_data_generator.steps_per_epoch(),
            steps_per_epoch=training_data_generator.steps_per_epoch(), 
            epochs=epochs, callbacks= self.callbacks)

    def predict(self, X):
        predictions = np.empty(shape=[0, self.output_size])
        missing_samples = 0

        samples_without_batch = X.shape[0] % self.batch_size
        
        if samples_without_batch > 0:
            missing_samples = self.batch_size - samples_without_batch
            missing_samples_shape = (missing_samples,) + (1,)*len(X[0].shape)
            complement = np.tile(X[0], missing_samples_shape)
            X = np.concatenate((X, complement), axis=0)

        batches_number = int(X.shape[0] / self.batch_size)

        X = X.reshape(-1, self.batch_size, X.shape[1], self.features_number)

        for x_batch in X:
            batch_predictions = self.model.predict(x_batch, batch_size = self.batch_size)
            predictions = np.concatenate((predictions, batch_predictions), axis = 0)
        
        if missing_samples > 0:
            return predictions[:-missing_samples,:]

        return predictions

    def evaluate(self, validation_data_generator):
        return self.model.evaluate(validation_data_generator)

    def load(self, model_dir):
        model_json_path = f'{model_dir}/{self.architecture_file_name}'
        model_weights_path = f'{model_dir}/{self.weights_file_name}'
        model_hyperparameters_path = f'{model_dir}/{self.hyperparameters_file_name}'

        # load json and create model
        with open(model_json_path, "r") as json_file:
            model_json = json_file.read()
            self.model = model_from_json(model_json)

        # load weights into new model
        self.model.load_weights(model_weights_path)
        print("Loaded model from disk")

        # load and return hyperparameters
        with open(model_hyperparameters_path, "r") as json_file:
            hyperparameters =  json.loads(json_file.read())
            self.__load_hyperparameters(hyperparameters)

            return hyperparameters


    def save(self, base_dir):
        model_dir = create_model_dir(base_dir)
        model_json_path = f'{model_dir}/{self.architecture_file_name}'
        model_weights_path = f'{model_dir}/{self.weights_file_name}'
        model_hyperparameters_path = f'{model_dir}/{self.hyperparameters_file_name}'

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_json_path, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(model_weights_path)

        # save model hyperparameters
        with open(model_hyperparameters_path, 'w') as file:
            hp = self.__get_hyper_parameters_dict()
            json.dump(hp, file)
        
        print(f'Saved model files to disk under{model_dir}')

    def __get_hyper_parameters_dict(self):
        loss_name = self.loss

        if not isinstance(self.loss, str):
            loss_name = self.loss.__name__
        
        return {
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'hidden_layer_size': self.hidden_layer_size,
            'lookback': self.lookback, 
            'loss': loss_name,
            'dropout_ratio': self.dropout_ratio,
            'features_number': self.features_number,
            'output_size': self.output_size,
            'pi_params': self.pi_params
        }

    def __load_hyperparameters(self, hyperparameters):
        try:

            self.epochs = hyperparameters['epochs']
            self.batch_size = hyperparameters['batch_size']
            self.hidden_layer_size = hyperparameters['hidden_layer_size']
            self.lookback = hyperparameters['lookback']
            self.dropout_ratio = hyperparameters['dropout_ratio']
            self.features_number = hyperparameters['features_number']
            self.output_size = hyperparameters['output_size']
            self.pi_params = hyperparameters['pi_params']
            self.learning_rate = hyperparameters['learning_rate']

        except Exception as e:
            print(f'Missing key{e}')

    