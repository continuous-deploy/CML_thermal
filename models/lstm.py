import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint

class LSTMModel:
    def __init__(self, input_shape:tuple, model_path:str="models/lstm_model.keras"):
        if os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
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

    def save_model(self, path="models/lstm_model.keras"):
        # Save the trained model
        self.model.save(path)
        print(f"Model saved at {path}")

    def evaluate_model(self, X, y, title:str, save_path:str="metrics/lstm/lstm_eval"):        
        # Evaluate past model on recent data
        past_evaluation = self.model.evaluate(X, y, verbose=0)
        y_pred = self.model.predict(X)

        # Plot actual vs predicted values for past model
        plt.figure(figsize=(10, 6))
        plt.plot(y, label='Actual value')
        plt.plot(y_pred, label='Predicted value')
        plt.title(f'Actual vs Predicted\n{title}')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(save_path + f"{title}" + f"{int(datetime.now().timestamp())}.png")

        print(f"LSTM Model Evaluation on Recent Data - Loss: {past_evaluation[0]}, MAE: {past_evaluation[1]}")
        
        return {"mse": past_evaluation[0], "mae": past_evaluation[1]}
