from flask import Flask, request, jsonify
import pandas as pd
import pickle
import datetime
from sklearn.preprocessing import  MinMaxScaler
from scipy.stats import zscore
import numpy as np



# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained random forrest model model using pickle
with open('models/random_forrest_model-24-09-2024-14-38-55-00.pkl', 'rb') as f:
    model = pickle.load(f)


def preprocess_input(df):
    def replace_outliers_with_mean(df, z_threshold=3):
        # Iterate through each numeric column
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['Sales','Customers']:
                col_data = df[col].dropna()
                col_zscore = zscore(col_data)
                
                # Create a boolean mask for outliers
                outlier_mask = abs(col_zscore) > z_threshold
                
                # Calculate the mean of non-outlier values (excluding NaNs)
                mean_value = col_data[~outlier_mask].mean()
                
                # Ensure the column is of float type to avoid dtype incompatibility
                df[col] = df[col].astype(float)
                
                # Replace outliers in the original DataFrame Need to align the original index with the calculated z-scores
                df.loc[df[col].notna() & (abs(zscore(df[col].fillna(0))) > z_threshold), col] = mean_value
        return df
    result1=replace_outliers_with_mean(df)
    def handle_missing_values(df):
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
    
    result2=handle_missing_values(result1)
    
    def feature_engineering(df):
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
    
    result3=feature_engineering(result2)

    
    def encode_categorical(df):
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
    
    result4=encode_categorical(result3)
            
    def scale_data(df):
        scaler = MinMaxScaler()
        
        columns_to_scale=['Sales', 'Customers','CompetitionDistance', 'CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','DaysToHoliday','DaysAfterHoliday']
        # Apply scaler only to the specified numeric columns
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        
        return df
    
    processed_df=scale_data(result4)
    processed_df = processed_df.drop('Sales', axis=1)
    
    return processed_df

# Define the prediction endpoint for CSV file input
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure that a CSV file is included in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        # Read the CSV file from the request
        file = request.files['file']
        input_data = pd.read_csv(file)

        # Preprocess the data
        processed_data = preprocess_input(input_data)
        # # Make sure to drop the 'Sales' column (since it's the target) if it exists
        # if 'Sales' in input_data.columns:
        #     input_data = input_data.drop('Sales', axis=1)

        # Make predictions using the loaded model
        predictions = model.predict(processed_data)

        # Return the predictions as a JSON response
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define a health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running', 'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
