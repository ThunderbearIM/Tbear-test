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
        return df