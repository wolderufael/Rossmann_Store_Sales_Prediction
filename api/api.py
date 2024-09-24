from flask import Flask, request, jsonify
import pandas as pd
import pickle
import datetime
from sklearn.preprocessing import  MinMaxScaler
from scipy.stats import zscore
import numpy as np
import sys
import os

# Add the project directory to sys.path
current_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.abspath(os.path.join(current_directory, '..'))  # Adjust '..' based on project structure
sys.path.append(project_directory)

sys.path.append(os.path.abspath('../models'))

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained random forrest model model using pickle
with open('../models/random_forrest_model-24-09-2024-14-38-55-00.pkl', 'rb') as f:
    rf_model = pickle.load(f)
    
# Load the pre-trained LSTM model model using pickle
with open('../models/LSTM_model-24-09-2024-22-30-39-00.pkl', 'rb') as f:
    lstm_model = pickle.load(f)

from script.data_preprocessing import Preprocessor
rf_processor=Preprocessor()

def preprocess_input_rf(df):
    result1=rf_processor.replace_outliers_with_mean(df)
    result2=rf_processor.handle_missing_values(result1)
    result3=rf_processor.feature_engineering(result2)
    result4=rf_processor.encode_categorical(result3)
    processed_df=rf_processor.scale_data(result4)
    processed_df = processed_df.drop('Sales', axis=1)
    
    return processed_df
def preprocess_input_lstm(df):
    
    
# Define random forrest prediction endpoint for CSV file input
@app.route('/predict_random_forrest', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        input_data = pd.read_csv(file)

        processed_data = preprocess_input_rf(input_data)

        predictions = rf_model.predict(processed_data)

        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# @app.route('/predict_lstm', methods=['POST'])
# def predict():
#     try:
#         # Ensure that a CSV file is included in the request
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file provided'}), 400

#         # Read the CSV file from the request
#         file = request.files['file']
#         input_data = pd.read_csv(file)

#         # Preprocess the data
#         processed_data = preprocess_input(input_data)
#         # # Make sure to drop the 'Sales' column (since it's the target) if it exists
#         # if 'Sales' in input_data.columns:
#         #     input_data = input_data.drop('Sales', axis=1)

#         # Make predictions using the loaded model
#         predictions = model.predict(processed_data)

#         # Return the predictions as a JSON response
#         return jsonify({'predictions': predictions.tolist()})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# Define a health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running', 'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
