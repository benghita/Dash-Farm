import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.metrics import r2_score
from IPython.display import clear_output
import json
import requests


class Predictor :
    def __init__(self, data_path = '/Users/ghita/Desktop/fm/projects/AGRO-Dashboard/P075-6 Months_MVCV_18June23.csv',
                model_path = '/Users/ghita/Desktop/fm/projects/AGRO-Dashboard/ml_tools/P075_24h_3h_CNNLSTM.h5',
                scaler1_path = '/Users/ghita/Desktop/fm/projects/AGRO-Dashboard/ml_tools/scaler1.pkl',
                scaler2_path = '/Users/ghita/Desktop/fm/projects/AGRO-Dashboard/ml_tools/scaler2.pkl'):
        
        self.url = "https://eu-west-2.aws.data.mongodb-api.com/app/data-rpqya/endpoint/data/v1/action/find"

        self.headers = {
            "apiKey": "nzyQOL6Q8MzrC7ReUgDVmiqVYcXh1o7SUdI4sEiVbjSx4TUrXz7tJVJlpV5lgPiV",
            "Content-Type": "application/ejson",
            "Accept": "application/json",
            }
        
        self.thresholds = [25.10530472, 472.3631897]
        
    def fetch(self, cv):
        if cv == 1 : 
            collection = "P075"
        else :
            collection = "P092"

        # Specify the fields to project in the "projection" field of the payload
        payload = json.dumps({
            "dataSource": "Cluster0",
            "database": "agrodaraa",
            "collection": collection,
            "sort": {"Date": -1},
            "limit": 1440,
            "projection": {"CV1": 1, "MV1": 1, "Date": 1}
        })

        # Make the POST request
        response = requests.request("POST", self.url, headers = self.headers, data = payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
        else:
            print(f"Request failed with status code {response.status_code}")
        
        # Extract the list of dictionaries
        documents_list = data['documents']

        # Create a DataFrame from the list of dictionaries
        return pd.DataFrame(documents_list)

    def CV_sammary (self, cv):
        data = df[[column]].copy()
        total_rows = len(data)
        thr = 5
        # Mark outliers
        outliers = (data < self.thresholds[column][0]) | (data > self.thresholds[column][1])
        # Calculate the percentage of outliers for this variable
        percentage_outliers = (outliers.sum() / len(data)) * 100

        # Count NaN and missing values
        missing_count = data[column].isnull().sum()
        percentage_missing_count = (missing_count.sum() / len(data)) * 100

        # Calculate the frequency of each unique value
        unique_values = data[column].value_counts()
            # Calculate the percentage of frozen values for this variable
        frozen_count = sum(unique_values[unique_values <= thr])
        percentage_frozen = (frozen_count / total_rows) * 100

        return percentage_outliers[column], percentage_missing_count, percentage_frozen
