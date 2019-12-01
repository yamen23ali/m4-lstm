import keras
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from keras.models import model_from_json



class M4Model(object):

    def __init__(self, hidden_layer_size=100, batch_size=50, lookback=48, 
        horizon=48, learning_rate=0.001, loss='mae'):

        self.batch_size = batch_size

        self.model = Sequential()

        self.model.add(LSTM(hidden_layer_size, batch_input_shape=(batch_size, lookback, 1),  return_sequences=True, activation='tanh',
              kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)))

        #self.model.add(LSTM(hidden_layer_size, batch_input_shape=(batch_size, lookback,1), return_sequences=True, activation='tanh',
        #      kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)))

        self.model.add(LSTM(hidden_layer_size, batch_input_shape=(batch_size, lookback,1),  activation='tanh',
              kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)))

        self.model.add(Dense(horizon, activation='linear',
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.3)))

        opt = optimizers.RMSprop(lr=learning_rate)#, clipvalue=0.3)
        #opt = optimizers.SGD(lr=0.01, decay=1e-2, momentum=0.7, nesterov=True)

        self.model.compile(loss=loss, optimizer=opt)

    def train(self, data_generator, epochs):
        hist = self.model.fit_generator(data_generator, steps_per_epoch= data_generator.__len__(), epochs=epochs)
        return hist

    def predict(self, X):
        return self.model.predict(X, batch_size = self.batch_size)

    def load(self, model_json_path, model_weights_path):
        # load json and create model
        json_file = open(model_json_path, 'r')
        model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(model_json)

        # load weights into new model
        self.model.load_weights(model_weights_path)
        print("Loaded model from disk")

    def save(self, model_json_path, model_weights_path):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_json_path, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(model_weights_path)
        print("Saved model to disk")
    