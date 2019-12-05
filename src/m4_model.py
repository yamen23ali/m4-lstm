import keras
import tensorflow as tf
import json

from src.utils import create_model_dir

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
from keras.models import model_from_json


class M4Model(object):

    def __init__(self, hidden_layer_size=100, batch_size=50, lookback=48, 
        horizon=48, learning_rate=0.001, loss='mae', dropout_ratio=0.0):

        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.lookback = lookback
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.loss = loss
        self.dropout_ratio = dropout_ratio

        self.model = Sequential()

        self.model.add(LSTM(hidden_layer_size, batch_input_shape=(batch_size, lookback,1), return_sequences=True, activation='tanh',
             kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2), recurrent_dropout=dropout_ratio))

        self.model.add(LSTM(hidden_layer_size, batch_input_shape=(batch_size, lookback,1),  activation='tanh',
              kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2), recurrent_dropout=dropout_ratio))

        self.model.add(Dense(horizon, activation='linear',
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.3)))

        self.opt = optimizers.RMSprop(lr=learning_rate)#, clipvalue=0.3)
        #opt = optimizers.SGD(lr=0.01, decay=1e-2, momentum=0.7, nesterov=True)

        self.model.compile(loss=self.loss, optimizer=self.opt)

    def __get_hyper_parameters_dict(self):
        loss_name = self.loss

        if not isinstance(self.loss, str):
            loss_name = self.loss.__name__
        
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'hidden_layer_size': self.hidden_layer_size,
            'lookback': self.lookback, 
            'loss': loss_name,
            'dropout_ratio': self.dropout_ratio
        }

    def compile(self):
        self.model.compile(loss=self.loss, optimizer=self.opt)

    def train(self, training_data_generator, test_data_generator, epochs):
        self.epochs = epochs

        return self.model.fit_generator(training_data_generator,
            validation_data = test_data_generator,
            validation_steps=test_data_generator.steps_per_epoch(),
            steps_per_epoch=training_data_generator.steps_per_epoch(), 
            epochs=epochs)

    def predict(self, X):
        return self.model.predict(X, batch_size = self.batch_size)

    def evaluate(self, validation_data_generator):
        return self.model.evaluate(validation_data_generator)

    def load(self, model_json_path, model_weights_path):
        # load json and create model
        json_file = open(model_json_path, 'r')
        model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(model_json)

        # load weights into new model
        self.model.load_weights(model_weights_path)
        print("Loaded model from disk")

    def save(self, base_dir):
        model_dir = create_model_dir(base_dir)
        model_json_path = f'{model_dir}/architecture.json'
        model_weights_path = f'{model_dir}/weights.h5'
        model_hyperparameters_path = f'{model_dir}/hyperparameters.json'


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
    