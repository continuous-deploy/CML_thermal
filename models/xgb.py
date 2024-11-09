import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List

class XGBoostModel:
    def __init__(self, n_estimators: int = 100, max_depth: int = 3, learning_rate: float = 0.1):
        # Initialize the model with given parameters
        self.model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

    def fit_model(self, X, y):
        # Train the model
        self.model.fit(X, y)
        
    def test(self, X_test, y_test, save_path="metrics/xgboost"):
        # Predict on test data and evaluate metrics
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Test MSE: {mse}, Test MAE: {mae}")
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(np.array(y_test), label='Actual Values', color='b')
        plt.plot(y_pred, label='Predicted Values', color='r')
        plt.xlabel('Samples')
        plt.ylabel('Values')
        plt.title(f"Actual vs Predicted Values\nMSE: {mse:.4f}, MAE: {mae:.4f}")
        plt.legend()

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, f'xgboost_plot_{np.round(mse, 1)}_{datetime.now().timestamp()}.png')

        # Save the plot
        plt.savefig(plot_path)
        print(f"Plot saved at {plot_path}")
        
        return mse

    def fit_and_evaluate(self, train_data, test_data, X_col:List, y_col:List):
        # Train the model on training data
        self.fit_model(train_data[X_col], train_data[y_col])

        # Evaluate the model on the test data
        mse = self.test(test_data[X_col], test_data[y_col])

        return mse

    def save_model(self, save_path="models/xgboost_model.pkl"):
        # Save the model to a file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.model, save_path)
        print(f"Model saved at {save_path}")

    @staticmethod
    def load_model(model_path="models/xgboost_model.pkl", n_estimators: int = 100, max_depth: int = 3, learning_rate: float = 0.1):
        # Check if the model file exists
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            return joblib.load(model_path)
        else:
            print(f"No saved model found at {model_path}. Creating a new XGBoostModel instance.")
            return XGBoostModel(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
