import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

class LSTMModel:
    def __init__(self, input_shape):
        self.model = Sequential()
        
        # Define the LSTM layers
        self.model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        self.model.add(LSTM(32, return_sequences=False))
        self.model.add(Dense(1, activation='linear'))
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    def summary(self):
        # Display the model summary
        return self.model.summary()
    
    def fit(self, X, y, epochs=50, batch_size: int = 32, validation_ratio:float = 0.2, save_temp=False):
        # Checkpoint to save model weights
        if save_temp:
            checkpoint = ModelCheckpoint("temp_lstm_model.h5", save_best_only=True, monitor="val_loss")
            history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_ratio, callbacks=[checkpoint])
        else:
            history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_ratio)
        return history

    def test(self, X_test, y_test):
        # Evaluate the model
        evaluation = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = self.model.predict(X_test)
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig("lstm_evaluation_plot.png")
        plt.show()

        return {"loss": evaluation[0], "mae": evaluation[1]}

    def save_model(self, path="models/lstm_model.keras"):
        # Save the trained model
        self.model.save(path)
        print(f"Model saved at {path}")

    def evaluate_past_model(self, recent_data_X, recent_data_y, temp_model_path="models/lstm_model.keras"):
        # Load the past model
        if not os.path.exists(temp_model_path):
            print("Past model not found for evaluation.")
            return
        
        past_model = load_model(temp_model_path)
        
        # Evaluate past model on recent data
        past_evaluation = past_model.evaluate(recent_data_X, recent_data_y, verbose=0)
        recent_y_pred = past_model.predict(recent_data_X)

        # Plot actual vs predicted values for past model
        plt.figure(figsize=(10, 6))
        plt.plot(recent_data_y, label='Actual')
        plt.plot(recent_y_pred, label='Past Model Predicted')
        plt.title('Past Model: Actual vs Predicted on Recent Data')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig("past_lstm_model_evaluation.png")

        print(f"Past Model Evaluation on Recent Data - Loss: {past_evaluation[0]}, MAE: {past_evaluation[1]}")
        
        return {"past_loss": past_evaluation[0], "past_mae": past_evaluation[1]}

    def retrain_and_evaluate(self, X_train, Y_train, X_test, y_test):
        