import pandas as pd
from datetime import timedelta
import requests
import json



class Preprocessor : 

    def __init__ (self) : 
        self.thresholds = {
            'MV1': (4.342303276, 488.2485352),
            'MV2': (4.300886154, 1237.251953),
            'MV3': (2.820113897, 562.4943237),
            'MV4': (-1.878349543, 258.3875122),
            'MV5': (18.381464, 757.949707),
            'MV6': (0, 300),
            'MV7': (2.225207329, 39.08103943),
            'MV8': (0.732448936, 39.08103943),
            'MV9': (1.5459584, 39.08103943),
            'MV10': (4.465659618, 106.5072937),
            'MV11': (2.27641654, 39.08103943),
            'MV12': (5.236536503, 114.5099945),
            'CV1': (25.10530472, 472.3631897)
        }
    
        self.url = "https://eu-west-2.aws.data.mongodb-api.com/app/data-rpqya/endpoint/data/v1/action/find"

        self.payload = json.dumps({
            "dataSource": "Cluster0",
            "database": "DB-P104",
            "collection": "SmartFarm",
            "sort": { "Date": -1 },
            "limit": 1440
        })
        self.headers = {
            "apiKey": "nzyQOL6Q8MzrC7ReUgDVmiqVYcXh1o7SUdI4sEiVbjSx4TUrXz7tJVJlpV5lgPiV",
            "Content-Type": "application/ejson",
            "Accept": "application/json",
        }

    def request_24(self):
        # Make the POST request
        response = requests.request("POST", self.url, headers = self.headers, data = self.payload)
                # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
        else:
            print(f"Request failed with status code {response.status_code}")
        # Extract the list of dictionaries
        documents_list = data['documents']

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(documents_list)
        df.drop("_id", axis=1, inplace= True)

        self.df = df[['Date', 'MV1', 'MV2', 'MV3', 'MV4', 'MV5', 'MV6', 'MV7', 'MV8', 'MV9', 'MV10', 'MV11', 'MV12', 'CV1']]

        self.df['Date'] = pd.to_datetime(self.df['Date'], format="%Y-%m-%dT%H:%M:%SZ")
        last_recorded_date = self.df['Date'].max()  # Get the maximum (most recent) date

        return df, last_recorded_date

    def summary(self, df):
        data = df.copy()
        # Create a dictionary to store the percentage of outliers, NaN, and missing values for each variable
        summary_dict = {'Variable': [], 'Outlier': [],  'Missing': [], 'Frozen': []}
        # Need to add, % frozen values
        total_rows = len(data)
        thr = 5
        # Loop through acceptance_ranges to mark outliers and calculate counts of NaN and missing values for each variable
        for column, (min_value, max_value) in self.thresholds.items():
            # Mark outliers
            outliers = (data[column] < min_value) | (data[column] > max_value)
            data[column + '_outlier_'] = outliers

            # Calculate the percentage of outliers for this variable
            percentage_outliers = (outliers.sum() / len(data)) * 100

            # Count NaN and missing values
            nan_count = data[column].isna().sum()
            percentage_nan_count = (nan_count.sum() / len(data)) * 100

            missing_count = data[column].isnull().sum()
            percentage_missing_count = (missing_count.sum() / len(data)) * 100

            # Calculate the frequency of each unique value
            unique_values = data[column].value_counts()
                # Calculate the percentage of frozen values for this variable
            frozen_count = sum(unique_values[unique_values <= thr])
            percentage_frozen = (frozen_count / total_rows) * 100

            # Append values to the summary dictionary
            summary_dict['Variable'].append(column)
            summary_dict['Outlier'].append(percentage_outliers)
            #summary_dict['percentage NaN Count (%)'].append(percentage_nan_count)
            summary_dict['Missing'].append(percentage_missing_count )
            summary_dict['Frozen'].append(percentage_frozen)

        # Create a DataFrame from the summary_dict
        summary_df = pd.DataFrame(summary_dict)
        return summary_df
        

