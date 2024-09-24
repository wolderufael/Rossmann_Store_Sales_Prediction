import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        self.evaluate_model(y_test, y_pred)
        
        return model

