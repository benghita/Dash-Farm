import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow as tf
import joblib
import json
import requests


class Predictor :

    def __init__(self, ml_tools_path = 'ml_tools/', num_cvs = 4):
        
        # Define the threshold of CV
        self.thresholds = [20, 450]

        # Define list of collection
        self.collections = ['P075', 'P092', 'MG2', 'MG3']

        # Define the URL and the header to connect to mongodb API
        self.url = "https://eu-west-2.aws.data.mongodb-api.com/app/data-rpqya/endpoint/data/v1/action/find"

        self.headers = {
            "apiKey": "nzyQOL6Q8MzrC7ReUgDVmiqVYcXh1o7SUdI4sEiVbjSx4TUrXz7tJVJlpV5lgPiV",
            "Content-Type": "application/ejson",
            "Accept": "application/json",
            }
        
        # Define ML tools (models and scalers) for each CV
        self.models = []
        self.scalers_1 = []
        self.scalers_2 = []
        for i in range(1, num_cvs+1) :
            model = tf.keras.models.load_model(ml_tools_path+'model_'+str(i)+'.h5')
            self.models.append(model)

            with open(ml_tools_path+'scaler1_'+str(i)+'.pkl', 'rb') as file1:
                self.scalers_1.append(joblib.load(file1))

            with open(ml_tools_path+'scaler2_'+str(i)+'.pkl', 'rb') as file2:
                self.scalers_2.append(joblib.load(file2))

        
    def fetch_data(self, collection):

        payload = json.dumps({
                "dataSource": "Cluster0",
                "database": "agrodaraa",
                "collection": collection,
                "sort": {"Date": -1},
                "limit": 1440,
                "projection": {"CV1": 1, "MV1": 1, "Date": 1, "predition": 1}
            })
        # Make the POST request
        response = requests.request("POST", self.url, headers = self.headers, data = payload)
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
        else:
            print(f"Request failed with status code {response.status_code}")

        return pd.DataFrame(data['documents'])
       
    def fetch(self):

        self.documents_list = []
        self.dfs = []

        for collection in self.collections :

            df = self.fetch_data(collection)
            self.dfs.append(df)

            # Fill null values with forward fill (ffill)
            #df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%dT%H:%M:%SZ")
            self.documents_list.append(df.fillna(method='ffill'))

        # Create a DataFrame from the list of dictionaries
        return self.documents_list
    
    def CV_sammary (self, cv):
        df = self.dfs[cv]
        data = df[['CV1']].copy()
        total_rows = len(data)
        thr = total_rows * 0.15
        # Mark outliers
        outliers = (data < self.thresholds[0]) | (data > self.thresholds[1])
        # Calculate the percentage of outliers for this variable
        percentage_outliers = (outliers.sum() / len(data)) * 100

        # Count NaN and missing values
        missing_count = data['CV1'].isnull().sum()
        percentage_missing_count = (missing_count.sum() / len(data)) * 100

        # Calculate the frequency of each unique value
        unique_values = data['CV1'].value_counts()
            # Calculate the percentage of frozen values for this variable
        frozen_count = sum(unique_values[unique_values >= thr])
        percentage_frozen = (frozen_count / total_rows) * 100

        return [percentage_frozen, percentage_missing_count, percentage_outliers['CV1']]
    
    def predict_3h(self, cv=4, features = ['MV1','CV1']):
        # Define list of predictions and list of saved predictions
        self.predictions = []

        for i in range(cv):

            # Define ML tools for each variabe 
            sc1 = self.scalers_1[i]
            sc2 = self.scalers_2[i]
            model = self.models[i]
            df = self.documents_list[i]

            # Last Record Date
            date = df[['Date']].max()

            # Prepare data
            X_test = df[features][:1440]
            #X_test=np.array(X_test, dtype=np.float32)
            X_test_sc = sc1.transform(X_test)
            X_test_sc = X_test_sc[np.newaxis, :, :]

            # Prediction : 
            y = model.predict(X_test_sc)
            y_pred_reshaped = y.reshape(-1, y.shape[-2])
            y_pred = sc2.inverse_transform(y_pred_reshaped)
            
            # Add column fot Date
            date_rng = pd.date_range(start=date['Date'], periods=180, freq='1min')
            # Create a DataFrame with the array and the DatetimeIndex
            df_pred = pd.DataFrame(y_pred.T, index=date_rng)

            # Rename the columns to 'date' and 'prediction'
            df_pred = df_pred.reset_index()
            df_pred.columns = ['Date', 'prediction']

            self.predictions.append(df_pred)

        return self.predictions

