import tensorflow as tf
import json
import numpy as np

from src.utils import create_model_dir
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json


class M4Model(object):

    """ 
        This class represent the Model that is trained on the m4 data

        The model mainly depends on sequential LSTM cells followed by a dense layer.
        Dorpout and clipping are used to prevent overfitting and exploding gradient.

        Args:
            hidden_layers (int): The number of hidden LSTM layers
            hidden_layer_size (int): The number of hidden units in each LSTM layer
            batch_size (int): The number of samples in one batch
            lookback (int): How many steps to lookback in the past (i.e. The input of the model)
            output_size (int): The size of the model output
            learning_rate(float): The learning rate of the RMSprop algorithm used in training
            loss(func): The loss function to use in the training
            dropout_ratio(float): The probability of dropping a hidden unit during training
            features_number(int): The number of features in the input data
            clipvalue(float): The value to clip the gradient at
            pi_params(dict): The parameters used in building the new features for KL divergence approach(only used when loading a trained model)
            callbacks(:obj:`list` of func): A list of functions to use as call backs during training



        Args:
            hidden_layers (int): The number of hidden LSTM layers
            hidden_layer_size (int): The number of hidden units in each LSTM layer
            batch_size (int): The number of samples in one batch
            lookback (int): How many steps to lookback in the past (i.e. The input of the model)
            output_size (int): The size of the model output
            learning_rate(float): The learning rate of the RMSprop algorithm used in training
            loss(func): The loss function to use in the training
            dropout_ratio(float): The probability of dropping a hidden unit during training
            features_number(int): The number of features in the input data
            clipvalue(float): The value to clip the gradient at
            pi_params(dict): The parameters used in building the new features for KL divergence approach(only used when loading a trained model)
            callbacks(:obj:`list` of func): A list of functions to use as call backs during training
            architecture_file_name(str): The name of the architecture json file for a saved model
            weights_file_name(str): The name of the weights json file for a saved model
            hyperparameters_file_name(str): The name of the hyperparameters json file for a saved model
    """

    def __init__(self, hidden_layers = 2, hidden_layer_size=100, batch_size=50, lookback=48, 
        output_size=48, learning_rate=0.001, loss='mae', dropout_ratio=0.0, features_number = 1, 
        clipvalue=3, pi_params={}, callbacks=[]):

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
        self.clipvalue = clipvalue
        self.pi_params = pi_params
        self.callbacks = callbacks
        self.hidden_layers = hidden_layers

        self.model = Sequential()

        for i in range(0, self.hidden_layers-1):
            self.model.add(LSTM(hidden_layer_size, batch_input_shape=(batch_size, lookback,features_number), return_sequences=True, activation='tanh',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.3), recurrent_dropout=dropout_ratio))

        self.model.add(LSTM(hidden_layer_size, batch_input_shape=(batch_size, lookback,features_number), activation='tanh',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.3), recurrent_dropout=dropout_ratio))

        self.model.add(Dense(output_size, activation='linear',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.2)))

        self.opt = optimizers.RMSprop(lr=learning_rate, clipvalue=clipvalue)

        self.model.compile(loss=self.loss, optimizer=self.opt)

    def compile(self):
        """
            Complie the model (used after laoding a saved model)
        """
        self.model.compile(loss=self.loss, optimizer=self.opt)

    def train(self, training_data_generator, test_data_generator, epochs):
        """
            Train the model on training data and test on test data after each epoch.

            Args:
                training_data_generator (:obj: `M4Generator`): A generator of training data that give batches of (input, target) timeseries
                test_data_generator (:obj: `M4Generator`): A generator of test data that give batches of (input, target) timeseries

            Returns:
                (:obj: `History`): Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        self.epochs = epochs

        return self.model.fit_generator(training_data_generator,
            validation_data = test_data_generator,
            validation_steps=test_data_generator.steps_per_epoch(),
            steps_per_epoch=training_data_generator.steps_per_epoch(), 
            epochs=epochs, callbacks= self.callbacks)

    def predict(self, X):
        """
            Predict the next (horizon) time steps of the input timeseries

            Args:
                X (array_like): The model input timeseries

            Returns:
                (array_like): The model predictions
        """
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

    def evaluate(self, holdout_data_generator):
        """
            Evaluate (i.e predict) the model on a data, this is used to turn off the dropout

            Args:
                holdout_data_generator (:obj: `M4Generator`): A generator of holdout data that give batches of (input, target) timeseries

            Returns:
                (array_like): The model predictions
        """
        return self.model.evaluate(holdout_data_generator)

    def load(self, model_dir):
        """
            Load a saved model

            Args:
                model_dir (str): The path that contains the saved model weights, archeticture and hyper parameters json files

            Returns:
                (dict): The model hyper parameters
        """
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
        """
            Save a model under a directory

            Args:
                base_dir (str): The path to save the model weights, archeticture and hyper parameters json files under
        """
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
        """
            Get the model hyper parameters dict

            Args:
                (dict): The model hyper parameters
        """
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
            'clipvalue': self.clipvalue,
            'output_size': self.output_size,
            'pi_params': self.pi_params,
            'hidden_layers': self.hidden_layers
        }

    def __load_hyperparameters(self, hyperparameters):
        """
            Load hyper parameters into the current model

            Returns:
                (dict): The model hyper parameters
        """
        try:

            self.epochs = hyperparameters['epochs']
            self.batch_size = hyperparameters['batch_size']
            self.hidden_layer_size = hyperparameters['hidden_layer_size']
            self.lookback = hyperparameters['lookback']
            self.dropout_ratio = hyperparameters['dropout_ratio']
            self.features_number = hyperparameters['features_number']
            self.output_size = hyperparameters['output_size']
            self.pi_params = hyperparameters['pi_params']
            self.clipvalue = hyperparameters['clipvalue']
            self.learning_rate = hyperparameters['learning_rate']
            self.hidden_layers = hyperparameters['hidden_layers']

        except Exception as e:
            print(f'Missing key{e}')

    