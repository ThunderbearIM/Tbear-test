import pandas as pd
import numpy as np
from numpy import array, hstack
from typing import Tuple
from sklearn.model_selection import train_test_split


class DataTreatmentTimeSeries:

    def __init__(self):
        pass

    def read_csv(self, path: str) -> pd.DataFrame():
        """
        Returns a pandas dataframe from a csv file
        :param path: path to csv file
        :return: pandas dataframe
        """
        df = pd.read_csv(path)
        df = df.dropna()
        df = df.set_index(df.columns[0])
        return df

    def average_by_index(self, df: pd.DataFrame()) -> pd.DataFrame():

        """
        Returns a dataframe with the average value for a column for each index value
        :return: avg_df
        """
        df = df.copy()
        df = df.groupby(df.index).mean().dropna()
        return df

    def to_array_X_and_y_1d(self, df: pd.DataFrame()):

        """
        Returns the index as X and the first column as y
        :param df: pandas dataframe
        :return X: numpy array
        :return y: numpy array
        """
        df = df.copy()
        X = df.reset_index().index.values
        y = df.iloc[:, 0].values

        return X, y

    @staticmethod
    def split_sequences(sequences, n_steps_in, n_steps_out):
        """
        Splits a sequence into X and y, courtesy of Jason Brownlee
        :param sequence:
        :param n_steps_in:
        :param n_steps_out:
        :return:
        """
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

