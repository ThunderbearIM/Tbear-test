import pandas as pd
import tensorflow
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten, Input, concatenate, BatchNormalization
from keras.models import Model
from keras import Sequential

class LSTM_additions:

    def __init__(self):
        pass

    @staticmethod
    def LSTM_model(input_shape, output_shape, units,  optimizer='adam', loss='mse', metrics=['mse'], ):
        """
        Returns a keras LSTM model
        :param units:
        :param input_shape: shape of the input
        :param output_shape:
        :param dropout: dropout rate
        :param recurrent_dropout: recurrent dropout rate
        :param optimizer: optimizer
        :param loss: loss function
        :param metrics: metrics
        :return: keras LSTM model
        """
        model = Sequential()
        model.add(LSTM(units=units, input_dim=input_shape, return_sequences=False))
        model.add(Dense(units=output_shape))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def LSTM_unit_optimization(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32, verbose=1):
        """
        Returns a dictionary of the optimal LSTM units for each column in the given dataframe
        after testing in a LSTM model
        :param df: dataframe
        :return: unit: unit with lowest mse
        """
        unit_dict = {}
        test_units = [10, 20, 30, 40, 50]
        for units in test_units:
            input_shape = (X_train.shape[1], X_train.shape[2])
            output = y_train.shape[1]
            model = self.LSTM_model(input_shape=input_shape,output_shape=output, units=units)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
            mse = model.evaluate(X_test, y_test, verbose=verbose)
            unit_dict[units] = mse

        unit = min(unit_dict, key=unit_dict.get)

        return unit

    def LSTM_batch_size_optimization(self, X_train, y_train, X_test, y_test, epochs = 10, units=20, verbose=1):
        """
        Returns a dictionary of the optimal LSTM batch size for each column in the given dataframe
        after testing in a LSTM model
        :param df: dataframe
        :return: batch_size: batch_size with lowest mse
        """
        batch_size_dict = {}
        batch_test = [10, 20, 30, 40, 50]
        for batch in batch_test:
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = self.LSTM_model(input_shape=input_shape,units=units)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch, verbose=verbose)
            mse = model.evaluate(X_test, y_test, verbose=verbose)
            batch_size_dict[batch] = mse

        batch_size = min(batch_size_dict, key=batch_size_dict.get)

        return batch_size

    def combine_batch_and_unit(self, X_train, y_train, X_test, y_test, df, epochs=10, verbose=1):
        """
        Returns a dictionary of the optimal LSTM batch size and units for each column in the given dataframe
        after testing in a LSTM model
        :param df: dataframe
        :return: batch_size: batch_size with lowest mse
        """
        batch_size_dict = {}
        batch_test = [10, 20, 30, 40, 50]
        unit_dict = {}
        test_units = [10, 20, 30, 40, 50]
        for column in df.columns:
            input_shape = (X_train.shape[1], X_train.shape[2])
            output_shape = (y_train.shape[1], y_train.shape[2])
            model = self.LSTM_model(input_shape=input_shape, output_shape=output_shape)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
            mse = model.evaluate(X_test, y_test, verbose=verbose)
            batch_and_unit_dict[column] = mse

        batch_and_unit = min(batch_and_unit_dict, key=batch_and_unit_dict.get)

        return batch_and_unit