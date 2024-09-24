import pandas as pd
import pickle
from datetime import datetime
import os
import logging
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LSTM:
    def isolate_time_series(self,df, target_col='Sales'):
        logging.info("isolate time series data")
        df = df[['Date', target_col]].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    def check_stationarity(self,series):
        logging.info("check if it is stationary or not")
        result = adfuller(series)
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        if result[1] <= 0.05:
            print("The time series is stationary.")
        else:
            print("The time series is not stationary.")
            
    def difference_data(self,series):
        logging.info('apply differencing to make it stationary')
        return series.diff().dropna()
    
    def plot_correlation(self,series):
        logging.info("plot ACF and PACF")
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_acf(series, ax=plt.gca())
        plt.subplot(122)
        plot_pacf(series, ax=plt.gca())
        plt.show()
        
    def create_supervised_data(self,series, window_size=1):
        logging.info("Convert the time series data into a supervised learning")
        df = pd.DataFrame(series)
        columns = [df.shift(i) for i in range(1, window_size+1)]
        columns.append(df)
        df_supervised = pd.concat(columns, axis=1)
        df_supervised.dropna(inplace=True)
        return df_supervised
    
    def scale_data(self,df):
        logging.info("scale the features")
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(df)
        return scaled_data, scaler
    
    def prepare_data_for_lstm(self,data, window_size=1):
        logging.info("prepare data for LSTM")
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size, :-1])
            y.append(data[i + window_size - 1, -1])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self,input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.LSTM(50, activation='relu')(inputs)
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def evaluate_model(self,X,y,lstm_model):
        logging.info("evaluate the model")
        y_pred = lstm_model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae=mean_absolute_error(y,y_pred)
        r2=r2_score(y,y_pred)
        
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mse}")
        print(f"R^2 Score: {r2}")
        
    def save_model_with_timestamp(self,model,name, folder_path='models/'):
        logging.info('Serializes and saves a trained model with a timestamp.')
        
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Generate timestamp in format dd-mm-yyyy-HH-MM-SS-00
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-00")
        
        # Create a filename with the timestamp
        filename = f'{folder_path}{name}-{timestamp}.pkl'
        
        # Save the model using pickle
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        
        print(f"Model saved as {filename}")
        return filename
    
    
    
    