import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class DifferentialLinReg:

    @staticmethod
    def diff_data(df: pd.DataFrame):
        df = df.copy()
        df = df.diff(1)
        return df

    def diff_lin_reg(self, df: pd.DataFrame, ts=False):

        df = df.copy()
        """
        Returns a sklearn Linear Regression model based on the values in the given Dataframe,
        where the first column is X and the second column is y
        If the dataframe is a timeseries, then the index is X, and the first column is y
        
        :param df: Dataframe with X and y values
        :param ts: Boolean, if the dataframe is a timeseriese
        :return: sklearn Linear Regression model
        """
        if ts:
            x = df.reset_index().iloc[:, 0].values()
            y = df.iloc[:, 0].values()

        else:
            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values

        model = LinearRegression().fit(X=x,
                                       y=y)
        return model

    @staticmethod
    def diff_lin_reg_predict(self, model: LinearRegression, prediction_range: numpy.array(), last_datapoint: float):

        """
        Returns a prediction based on the given model and last datapoint
        We add together each point of differentials to the last datapoint, where each addition is a new prediction step
        Then transforms the prediction back from differentials to the original scale

        :param model: sklearn Linear Regression model
        :param prediction_range: range of predictions to make
        :param last_datapoint:
        :return: final_prediction
        """

        prediction = model.predict(prediction_range)
        prediction= np.append(last_datapoint, prediction)
        final_prediction = np.cumsum(prediction)

        return final_prediction

