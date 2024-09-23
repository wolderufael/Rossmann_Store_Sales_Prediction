import pandas as pd
import logging
from scipy.stats import zscore
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    def handle_missing_values(self,df):
        logging.info('handle missing values')
        # Fill missing CompetitionDistance with a new category 
        df.fillna({
            'CompetitionDistance': -1,  # Can also use -1 to mark missing distances
            'CompetitionOpenSinceMonth': 0,
            'CompetitionOpenSinceYear': 0,
            'Promo2SinceWeek': 0,
            'Promo2SinceYear': 0,
            'PromoInterval': 'None'
        }, inplace=True)
        
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

     