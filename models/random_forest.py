import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

class RandomForestModel:
    def __init__(self, n_estimators: int = 100, max_depth: int = None, model_path:str="models/rf_model.pkl"):
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            # Initialize the model with the given parameters
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    def summary(self):
        return self.model.summary
    
    def fit_model(self, X, y):
        # Train the model
        self.model.fit(X, y)
        
    def evaluate(self, X, y, tag:str, save_path="metrics/random_forest"):
        # # Ensure the model is fitted before making predictions
        # try:
        #     check_is_fitted(self.model)
        # except NotFittedError as e:
        #     raise RuntimeError("Model must be fitted before calling test. Call fit_model() first.") from e

        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        print(f"Test MSE: {mse}, Test MAE: {mae}")
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(np.array(y), label='Actual Values', color='b')
        plt.plot(y_pred, label='Predicted Values', color='r')
        plt.xlabel('Samples')
        plt.ylabel('Values')
        plt.title(f"Actual vs Predicted Values\nMSE: {mse:.4f}, MAE: {mae:.4f}: "+ tag)
        plt.legend()

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, f'rf_eval_{tag}_{int(datetime.now().timestamp())}.png')

        # Save the plot
        plt.savefig(plot_path)
        print(f"Plot saved at {plot_path}")
        
        return mse

    def save_model(self, save_path="models/rf_model.pkl"):
        # Save the model to a file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.model, save_path)
        print(f"Model saved at {save_path}")