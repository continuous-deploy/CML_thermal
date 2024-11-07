import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib  # For saving the model

class RandomForestModel:
    def __init__(self, n_estimators: int = 100, max_depth: int = None):
        # Initialize the model with the given parameters
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    def fit_model(self, X, y):
        # Train the model
        self.model.fit(X, y)
        
    def test(self, X_test, y_test, save_path="metrics/random_forest"):
        # Predict on test data and evaluate metrics
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Test MSE: {mse}, Test MAE: {mae}")
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        metrics_path = os.path.join(save_path, f'rf_mse_{np.round(mse,1)}_{datetime.now().timestamp()}.txt')
        
        # Save metrics to a file
        with open(metrics_path, 'w') as f:
            f.write(f"MSE: {mse}\nMAE: {mae}\n")
        print(f"Metrics saved at {metrics_path}")
        
        return mse

    def fit_and_evaluate(self, train_data, test_data, X_col, y_col):
        # Train the model on training data
        self.fit_model(train_data[X_col], train_data[y_col])

        # Evaluate the model on the test data
        mse = self.test(test_data[X_col], test_data[y_col])

        return mse

    def save_model(self, save_path="models/random_forest_model.pkl"):
        # Save the model to a file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.model, save_path)
        print(f"Model saved at {save_path}")

    @staticmethod
    def load_model(model_path="models/random_forest_model.pkl"):
        # Load and return the model from a file
        return joblib.load(model_path)
