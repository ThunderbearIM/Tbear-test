from pyjstat import pyjstat
import requests

class ApiRequest:

    """
    Class for making API requests to SSB
    """
    @staticmethod
    def call_ssb_to_df(POST_URL=str, payload=dict):
        result = requests.post(POST_URL, json=payload)
        dataset = pyjstat.Dataset.read(result.text)
        df = dataset.write('dataframe')

        return df