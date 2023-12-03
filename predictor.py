import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow as tf
import joblib
import json
import requests


class Predictor :

    def __init__(self, ml_tools_path = 'ml_tools/'):
        
        # Define the threshold of CV
        self.thresholds = [20, 450]

        # Define list of collection 'P075', 'P092', 'MG2', 'MG3', 
        self.collections = ['MG2_5CV']

        # Define the URL and the header to connect to mongodb API
        self.url = "https://eu-west-2.aws.data.mongodb-api.com/app/data-rpqya/endpoint/data/v1/action/find"

        self.headers = {
            "apiKey": "nzyQOL6Q8MzrC7ReUgDVmiqVYcXh1o7SUdI4sEiVbjSx4TUrXz7tJVJlpV5lgPiV",
            "Content-Type": "application/ejson",
            "Accept": "application/json",
            }
        
        # Define ML tools (models and scalers) for each CV
        self.models = []
        self.scalers = []

        model = tf.keras.models.load_model(ml_tools_path+'model.h5')
        self.models.append(model)

        with open(ml_tools_path+'scaler.pkl', 'rb') as file1:
            self.scalers.append(joblib.load(file1))

        
    def fetch_data(self, collection,):
        # projection = {"CV1": 1, "MV1": 1, "Date": 1, "prediction_CV1": 1}
        payload = json.dumps({
                "dataSource": "Cluster0",
                "database": "agrodaraa",
                "collection": collection,
                "sort": {"Date": -1},
                "limit": 1440,
                #"projection": projection
            })
        
        # Make the POST request
        response = requests.request("POST", self.url, headers = self.headers, data = payload)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
        else:
            print(f"Request failed with status code {response.status_code}")

        return pd.DataFrame(data['documents'])
       
    def prepare_data(self):

        self.documents_list = []
        self.dfs = []

        for collection in self.collections :

            df = self.fetch_data(collection)
            self.dfs.append(df)

            # Fill null values with forward fill (ffill)
            self.documents_list.append(df.fillna(method='ffill'))

        # Create a DataFrame from the list of dictionaries
        return self.documents_list
    
    def CV_sammary (self, df_num, col_name='CV1'):
        df = self.dfs[df_num]
        data = df[[col_name]].copy()
        total_rows = len(data)
        thr = total_rows * 0.15

        # Mark outliers
        outliers = (data < self.thresholds[0]) | (data > self.thresholds[1])
        # Calculate the percentage of outliers for this variable
        percentage_outliers = (outliers.sum() / len(data)) * 100

        # Count NaN and missing values
        missing_count = data[col_name].isnull().sum()
        percentage_missing_count = (missing_count.sum() / len(data)) * 100

        # Calculate the frequency of each unique value
        unique_values = data[col_name].value_counts()
        
        # Calculate the percentage of frozen values for this variable
        frozen_count = sum(unique_values[unique_values >= thr])
        percentage_frozen = (frozen_count / total_rows) * 100

        return [percentage_frozen, percentage_missing_count, percentage_outliers[col_name]]
    
    def predict_3h(self, model=0, features = ['CV1','CV2','CV3','CV4','CV5']):

        # Define list of predictions and list of saved predictions
        self.predictions = []

        df_pred = self.predict(model, features = features)

        # Rename the columns to 'date' and 'prediction'
        df_pred = df_pred.reset_index()
        columns = ['Date']
        for i in range(len(features)):
            columns.append(features[i])
        df_pred.columns = columns

        self.predictions.append(df_pred)

        return self.predictions
    
    def predict(self, i, features = ['MV1','CV1'],):
        # Define ML tools for each variabe 
        scaler = self.scalers[i]
        model = self.models[i]
        df = self.documents_list[i]

        # Last Record Date
        date = df[['Date']].max()

        # Prepare data
        X_test = df[features][:1440]
        X_test = scaler.transform(X_test)
        X_test = X_test[np.newaxis, :, :]

        # Prediction : 
        y = model.predict(X_test)
        y_pred = scaler.inverse_transform(y.reshape(y.shape[-2], -1))
        
        # Add column fot Date
        date_rng = pd.date_range(start=date['Date'], periods=1440, freq='1min')
        
        # Create a DataFrame with the array and the DatetimeIndex
        df_pred = pd.DataFrame(y_pred, index=date_rng)
        return df_pred