import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List

class XGBoostModel:
    def __init__(self, n_estimators:int = 100, max_depth:int = 3, learning_rate:float = 0.1, model_path:str = "models/xgboost_model.pkl"):
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            # Initialize the model with given parameters
            self.model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

    def fit_model(self, X, y):
        # Train the model
        self.model.fit(X, y)
        
    def evaluate(self, X, y, title:str, save_path="metrics/xgboost"):
        # Predict on test data and evaluate metrics
        y_pred = self.model.predict(X)
        mse = mean_squared_error(X, y)
        mae = mean_absolute_error(y, y_pred)
        print(f"Test MSE: {mse}, Test MAE: {mae}")
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(np.array(y), label='Actual Values', color='b')
        plt.plot(y_pred, label='Predicted Values', color='r')
        plt.xlabel('Samples')
        plt.ylabel('Values')
        plt.title(f"Actual vs Predicted Values\nMSE: {mse:.4f}, MAE: {mae:.4f}: "+title)
        plt.legend()

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, f'xgb_eval_{title}_{int(datetime.now().timestamp())}.png')

        # Save the plot
        plt.savefig(plot_path)
        print(f"Plot saved at {plot_path}")
        
        return mse

    def save_model(self, save_path="models/xgboost_model.pkl"):
        # Save the model to a file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.model, save_path)
        print(f"Model saved at {save_path}")
