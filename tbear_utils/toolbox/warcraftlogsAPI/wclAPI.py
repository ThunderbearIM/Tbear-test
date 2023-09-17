import requests
import pandas as pd
import numpy as np

class WCLAPI:
    client_ID = "9a283e8f-a830-4c73-b9f4-a2dc64aa152e"
    client_Secret = "DabWRDnnFgzOdOz0JyopTbOmNVIjGw0psA0uMpJb"
    # def __init__(self):
    @staticmethod
    def fetch_raid_data():

        # API endpoint for raid rankings
        url = "https://classic.warcraftlogs.com/v1/rankings/encounter/1000?zone=1000&api_key=98fda3b2-1fa8-43f5-9b10-160432900287&guild=The%20Middle&server=Golemagg"
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


if __name__ == "__WCLAPI__":
    WCLAPI()
