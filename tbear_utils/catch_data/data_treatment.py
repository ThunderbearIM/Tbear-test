import pandas as pd
import numpy as np


class DataTreatment:

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
        df = df.groupby(df.index).mean()
        return df