from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

class LSTM_Model:
    def __init__(self, input_shape):
        """
        Initializes an LSTM model for regression.
        
        Parameters:
        - input_shape: tuple, the shape of input data (timesteps, features)
        """
        self.model = Sequential()
        
        self.model.add(LSTM(64, input_shape=input_shape, return_sequences=False))

        self.model.add(Dense(32, activation='relu'))
        
        self.model.add(Dense(1, activation='linear'))  # Linear activation for regression
        
        # Compile the model
        self.model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

    def summary(self):
        return self.model.summary()
    
    def fit(self, X, y, epochs=50, batch_size=32, validation_ratio:float = 0.2):
        """
        Trains the LSTM model.
        
        Parameters:
        - X: array-like, input features (must be 3D for LSTM: [samples, timesteps, features])
        - y: array-like, target values
        - epochs: int, number of epochs to train
        - batch_size: int, size of each batch for training
        - validation_rati:  ratio of split between train and validation data
        
        Returns:
        - history: Training history object with loss and metrics
        """
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_ratio)
        return history
