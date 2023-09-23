import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class DifferentialLinReg:

    @staticmethod
    def diff_data(df: pd.DataFrame):
        df = df.copy()
        df = df.diff(1).dropna()
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
            x = df.reset_index().iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, 0].values.reshape(-1, 1)

        else:
            x = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, 1].values.reshape(-1, 1)

        model = LinearRegression().fit(X=x,
                                       y=y)
        return model

    @staticmethod
    def diff_lin_reg_predict(model: LinearRegression, prediction_range: np.array, last_datapoint: float):

        """
        Returns a prediction based on the given model and last datapoint
        We add together each point of differentials to the last datapoint, where each addition is a new prediction step
        Then transforms the prediction back from differentials to the original scale

        :param model: sklearn Linear Regression model
        :param prediction_range: range of predictions to make
        :param last_datapoint:
        :return: final_prediction
        """

        prediction_range = prediction_range.reshape(-1, 1)
        prediction = model.predict(prediction_range)
        prediction= np.append(last_datapoint, prediction)
        final_prediction = np.cumsum(prediction)

        return final_prediction

    def combine_diff_model_predict(self, df: pd.DataFrame, last_datapoint: float, ts = False):

        """
        Combines the differential, treatment and prediction of the original data to instantly give out a prediction

        :param df:
        :param last_datapoint:
        :param ts:
        :return:
        """

        df = df.copy()
        df = self.diff_data(df)
        model = self.diff_lin_reg(df, ts=True)
        prediction_range = np.array(range(0, len(df)))
        prediction = self.diff_lin_reg_predict(model=model,
                                               prediction_range=prediction_range,
                                               last_datapoint=last_datapoint)

        return prediction

