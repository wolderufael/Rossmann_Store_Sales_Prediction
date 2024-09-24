import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class Modelling:
    # Function to create a Random Forest model
    def create_random_forest_model(self):
        logging.info('Creates and returns a RandomForestRegressor model.')

        # Initialize RandomForestRegressor
        model = RandomForestRegressor(random_state=42)
        return model

    # Function to evaluate the model
    def evaluate_model(self,y_true, y_pred):
        logging.info('Evaluates the model')

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'R^2 Score: {r2}')

    # Full pipeline: Model Training and Evaluation
    def full_pipeline(self,df, target_column):
        logging.info('Full pipeline to train a RandomForestRegressor and evaluate it.')

        # Split the data into features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create the model
        model = self.create_random_forest_model()
        
        # Train the model
        model.fit(X_train, y_train)
        
        #featurenimportance
        importance = model.feature_importances_
        feature_names = X.columns 
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        self.evaluate_model(y_test, y_pred)
        
        return X,y,model,importance,feature_names

    def get_feature_importance(self,X,importance,feature_names):
        logging.info('feature importance')

        # Create a DataFrame for visualization
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        # Plot the feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title("Feature Importance from Random Forest")
        plt.show()
        
        return feature_importance_df
    
    
    def bootstrap_confidence_interval(self,X, y, model, n_iterations=100, ci=95):
        logging.info('Estimate bootstrap confidence interval')
        predictions = []

        # Perform bootstrapping
        for i in range(n_iterations):
            # Resample the data
            X_resampled, y_resampled = resample(X, y, replace=True, random_state=i)
            
            # Fit the model on the resampled data
            model.fit(X_resampled, y_resampled)
            
            # Make predictions on the original data
            y_pred = model.predict(X)
            predictions.append(y_pred)

        # Convert list to NumPy array for easier manipulation
        predictions = np.array(predictions)
        
        # Calculate the mean prediction
        prediction_mean = predictions.mean(axis=0)
        
        # Calculate confidence interval bounds
        lower_bound = np.percentile(predictions, (100 - ci) / 2.0, axis=0)
        upper_bound = np.percentile(predictions, 100 - (100 - ci) / 2.0, axis=0)
        
        print("Lower bound of the confidence interval: ",lower_bound)
        print("Upper bound of the confidence interval: ",upper_bound)
        print("Mean prediction across bootstrap samples: ",prediction_mean)
    
    def save_model_with_timestamp(model, folder_path='models/'):
        logging.info('Serializes and saves a trained model with a timestamp.')
        
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Generate timestamp in format dd-mm-yyyy-HH-MM-SS-00
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-00")
        
        # Create a filename with the timestamp
        filename = f'{folder_path}model-{timestamp}.pkl'
        
        # Save the model using pickle
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        
        print(f"Model saved as {filename}")
        return filename