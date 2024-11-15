import pandas as pd
import numpy as np
from models.ann import ANN
from models.lstm import LSTMModel
from models.random_forest import RandomForestModel
from models.xgb import XGBoostModel

from utils.preprocess_data import load_and_concat_data, save_past_dependence_merged_data
from utils.preprocess_data import X_col, y_col
from utils.make_report import create_report

# prepare data for model
load_and_concat_data()
save_past_dependence_merged_data()


# load data for training
training_data = pd.read_csv('temp/training_data_simple.csv')
test_data = pd.read_csv('temp/test_data_simple.csv')



# Model 1: ANN
ann = ANN()

# Evaluation of old model on recent data
ann_old_mse = ann.evaluate(test_data[X_col], test_data[y_col], tag="old")

# Retaining of ANN model with recent data
hist = ann.fit_model(training_data[X_col], training_data[y_col])

# Evaluation of new model on recent data
ann_old_mse = ann.evaluate(test_data[X_col], test_data[y_col], tag="new")

# Saving of latest model
ann.save_model()




# Model 2: Random Forest
rf_model = RandomForestModel()

# Evaluate the old model
old_rf_mse = rf_model.evaluate(test_data[X_col], test_data[y_col], tag="old")

# Fit and evaluate the new model
new_rf_mse = rf_model.fit_model(training_data[X_col], training_data[y_col])

# Evaluate the new model
old_rf_mse = rf_model.evaluate(test_data[X_col], test_data[y_col], tag="new")

# Save the new model
rf_model.save_model()



# Model 3: XGBoost
xgboost_model = XGBoostModel()

# Evaluate old model performance on new data
xgboost_model.evaluate(test_data[X_col], test_data[y_col], tag="old")

# Train and evaluate the model by retraining it
xgb_mse = xgboost_model.fit_model(training_data[X_col], training_data[y_col])

# Evaluate old model performance on new data
xgboost_model.evaluate(test_data[X_col], test_data[y_col], tag="new")

# Save the model
xgboost_model.save_model()



# Loading data for LSTM, timeseries model
training_data = np.load("temp/timedependent_train_compressed.npz")
test_data = np.load("temp/timedependent_test_compressed.npz") 



# Model 4: LSTM
trainX, train_y = training_data['X'], training_data['y']
testX, test_y = test_data['X'], test_data['y']

m,n,h = trainX.shape

print(trainX.shape, testX.shape)

lstm_model = LSTMModel(input_shape=(n,h))

lstm_old_mse = lstm_model.evaluate_model(testX, test_y, "old_model")

lstm_model.fit(trainX, train_y)

lstm_new_mse = lstm_model.evaluate_model(testX, test_y, "new_model")

lstm_model.save_model()





create_report()