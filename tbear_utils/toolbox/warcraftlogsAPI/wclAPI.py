import requests
import pandas as pd
import numpy as np

class wclAPI:

    # def __init__(self):

    @staticmethod
    def fetch_raid_data():

        # API endpoint for raid rankings
        url = "https://classic.warcraftlogs.com/v1/rankings/encounter/1000?zone=1000&api_key=YOUR_API_KEY&guild=The%20Middle&server=Golemagg"
        data = None
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as e:
            print("Failed to fetch raid data:", str(e))

        return data

    def main(self):
        # Fetch raid data from the API
        raid_data = self.fetch_raid_data()

        if raid_data:
            # Process the fetched raid data
            print("Fetched raid data:", raid_data)
            # Add your code to process the raid data here


if __name__ == "__main__":
    main()
