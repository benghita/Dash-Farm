import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow as tf
import joblib
import json
import requests
import pandas as pd
import ast

class Predictor :

    def __init__(self, ml_tools_path = 'ml_tools/'):
        
        # Define the threshold of CV
        self.thresholds = [20, 450]

        # Define list of collection 'P075', 'P092', 'MG2', 'MG3', 
        self.ref = pd.read_csv(f'{ml_tools_path}ref.csv', sep=";")

        # Define the URL and the header to connect to mongodb API
        self.url = "https://eu-west-2.aws.data.mongodb-api.com/app/data-rpqya/endpoint/data/v1/action/find"

        self.headers = {
            "apiKey": "nzyQOL6Q8MzrC7ReUgDVmiqVYcXh1o7SUdI4sEiVbjSx4TUrXz7tJVJlpV5lgPiV",
            "Content-Type": "application/ejson",
            "Accept": "application/json",
            }
        
        # Define ML tools (models and scalers) for each CV
        self.models = []
        self.in_scalers = []
        self.out_scalers = []

        for index, row in self.ref.iterrows():

            model = tf.keras.models.load_model(f'{ml_tools_path}'+row['model'])
            self.models.append(model)

            with open(f'{ml_tools_path}'+row['scaler_in'], 'rb') as file1:
                self.in_scalers.append(joblib.load(file1))

            with open(f'{ml_tools_path}'+row['scaler_out'], 'rb') as file1:
                self.out_scalers.append(joblib.load(file1))
        
    def fetch_data(self, dataSource, database, collection,):
        # projection = {"CV1": 1, "MV1": 1, "Date": 1, "prediction_CV1": 1}
        payload = json.dumps({
                "dataSource": dataSource,
                "database": database,
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

        for index, row in self.ref.iterrows():

            df = self.fetch_data(row['dataSource'], row['database'], row['collection'])
            
            # Get a list of numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            # Fill null values with forward fill (ffill)
            self.documents_list.append(df.fillna(method='ffill'))
            
            # Convert only the numeric columns to 'float16'
            df[numeric_cols] = df[numeric_cols].astype('float16')
            self.dfs.append(df)

        # Create a DataFrame from the list of dictionaries
        return self.documents_list
    
    def CV_sammary (self, row_num, col_name='CV1'):
        df = self.dfs[row_num]
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
    
    def make_predictions(self):

        # Define list of predictions and list of saved predictions
        self.predictions = []

        for index, row in self.ref.iterrows():
            df_pred = self.predict_3h(index, features = ast.literal_eval(row['features_in']))

            # Rename the columns to 'date' and 'prediction'
            df_pred = df_pred.reset_index()
            columns = ['Date']
            cols = ast.literal_eval(row['features_out'])
            for i in range(len(cols)):
                columns.append(cols[i])
            df_pred.columns = columns

            self.predictions.append(df_pred)

        return self.predictions
    
    def predict_3h(self, i, features):
        # Define ML tools for each variabe 
        in_scaler = self.in_scalers[i]
        out_scaler = self.out_scalers[i]
        model = self.models[i]
        df = self.documents_list[i]

        # Last Record Date
        date = df[['Date']].max()

        # Prepare data
        X_test = df[features][-720:]
        X_test = in_scaler.transform(X_test)
        X_test = X_test[np.newaxis, :, :]

        # Prediction : 
        y = model.predict(X_test)
        y_pred = out_scaler.inverse_transform(y.reshape(y.shape[-2], -1))
        
        # Add column fot Date
        date_rng = pd.date_range(start=date['Date'], periods=180, freq='1min')
        
        # Create a DataFrame with the array and the DatetimeIndex
        df_pred = pd.DataFrame(y_pred, index=date_rng)
        return df_pred.astype('float16')