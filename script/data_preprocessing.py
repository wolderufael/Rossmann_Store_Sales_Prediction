import pandas as pd
import logging
from scipy.stats import zscore
import numpy as np
from sklearn.preprocessing import  MinMaxScaler
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocessor:
    def replace_outliers_with_mean(self,data, z_threshold=3):
        logging.info("replace outliers of Sales and Customer")
        # Iterate through each numeric column
        for col in data.select_dtypes(include=[np.number]).columns:
            if col not in ['Sales','Customers']:
                col_data = data[col].dropna()
                col_zscore = zscore(col_data)
                
                # Create a boolean mask for outliers
                outlier_mask = abs(col_zscore) > z_threshold
                
                # Calculate the mean of non-outlier values (excluding NaNs)
                mean_value = col_data[~outlier_mask].mean()
                
                # Ensure the column is of float type to avoid dtype incompatibility
                data[col] = data[col].astype(float)
                
                # Replace outliers in the original DataFrame Need to align the original index with the calculated z-scores
                data.loc[data[col].notna() & (abs(zscore(data[col].fillna(0))) > z_threshold), col] = mean_value
        return data 
    
    def handle_missing_values(self,df):
        logging.info('handle missing values')
        # Fill missing CompetitionDistance with a new category 
        df.fillna({
            'CompetitionDistance': df['CompetitionDistance'].median(),  # Can also use  to mark missing distances
            'CompetitionOpenSinceMonth': 0,
            'CompetitionOpenSinceYear': 0,
            'Promo2SinceWeek': 0,
            'Promo2SinceYear': 0,
            'PromoInterval': 'None'
        }, inplace=True)
        
        return df
        
    def feature_engineering(self,df):
        logging.info('feature engineering')
        # Convert 'Date' to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract basic date-based features
        df['Month'] = df['Date'].dt.month
        df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x > 5 else 0)
        
            # Extract additional date-based features
        df['IsBeginningOfMonth'] =df['Date'].dt.day.apply(lambda x: 1 if x <= 10 else 0)
        df['IsMidMonth'] =df['Date'].dt.day.apply(lambda x: 1 if 10 < x <= 20 else 0)
        df['IsEndOfMonth'] =df['Date'].dt.day.apply(lambda x: 1 if x > 20 else 0)

        # Add holiday proximity features
        df['StateHoliday'] = df['StateHoliday'].map({'0': 0, 'a': 1, 'b': 1, 'c': 1})
        holiday_dates = df.loc[df['StateHoliday'] == '1', 'Date']
        df['DaysToHoliday'] = df.apply(lambda row: 0 if row['StateHoliday'] == '1' 
                               else (holiday_dates[holiday_dates > row['Date']].min() - row['Date']).days 
                               if (holiday_dates > row['Date']).any() 
                               else 0, axis=1)
        df['DaysAfterHoliday'] = df.apply(lambda row: 0 if row['StateHoliday'] == '1' 
                                  else (row['Date'] - holiday_dates[holiday_dates < row['Date']].max()).days 
                                  if (holiday_dates < row['Date']).any() 
                                  else 0, axis=1)
        # Drop original 'Date' column after feature extraction
        df.drop(columns=['Date'], inplace=True)

        return df

    def encode_categorical(self,df):
        #Sine and Cosine transformation for 'DayOfWeek' and 'Month'
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7) 
        df.drop('DayOfWeek', axis=1, inplace=True)
        
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12) 
        df.drop('Month', axis=1, inplace=True)
        #oneHot encoding for assortment,store type and promointerval
        df = pd.get_dummies(df, columns=['StoreType', 'Assortment', 'PromoInterval'], drop_first=False)
        # Ensure 'StoreType' is actually dropped if present
        if 'StoreType' in df.columns:
            df.drop('StoreType', axis=1, inplace=True)
        
        return df
        
        
    def scale_data(self,df):
        scaler = MinMaxScaler()
        
        columns_to_scale=['Sales', 'Customers','CompetitionDistance', 'CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','DaysToHoliday','DaysAfterHoliday']
        # Apply scaler only to the specified numeric columns
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        
        return df