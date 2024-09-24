import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTM:
# Assuming your preprocessed data is in 'train_df'
    def isolate_time_series(self,df, target_col='Sales'):
        df = df[['Date', target_col]].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    def check_stationarity(self,series):
        result = adfuller(series)
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        if result[1] <= 0.05:
            print("The time series is stationary.")
        else:
            print("The time series is not stationary.")
            
    def difference_data(self,series):
        return series.diff().dropna()
    
    def plot_correlation(self,series):
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_acf(series, ax=plt.gca())
        plt.subplot(122)
        plot_pacf(series, ax=plt.gca())
        plt.show()
        
    def create_supervised_data(self,series, window_size=1):
        df = pd.DataFrame(series)
        columns = [df.shift(i) for i in range(1, window_size+1)]
        columns.append(df)
        df_supervised = pd.concat(columns, axis=1)
        df_supervised.dropna(inplace=True)
        return df_supervised
    
    def scale_data(self,df):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(df)
        return scaled_data, scaler
    
    def prepare_data_for_lstm(self,data, window_size=1):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size, :-1])
            y.append(data[i + window_size - 1, -1])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self,input_shape):
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=input_shape))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def evaluate_model(self,X,y,lstm_model):
        y_pred = lstm_model.predict(X)
        mse = mean_squared_error(y, y_pred)
        print(f"Mean Squared Error: {mse}")
    
    
    
    