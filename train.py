"""
This file will trigger model retrainning after importingg all the models from model folder

As this file is triggered, It should do folowing task:
1. 
"""

import pandas as pd

from models.ann import ANN, old_ann_model
from models.lstm import LSTM
from models.random_forest import RandomForestModel
from models.xgb import XGBoostModel

from utils.preprocess_data import load_and_concat_data, save_past_dependence_merged_data
from utils.preprocess_data import X_col, y_col


load_and_concat_data()
save_past_dependence_merged_data()

training_data = pd.read_csv('temp/training_data_simple.csv')
test_data = pd.read_csv('temp/test_data_simple.csv')


# ann = ANN()
# ann_old = old_ann_model()

# old_ann_mse = ann_old.evaluate(test_data)

# _, new_ann_mse = ann.fit_and_evaluate(training_data, test_data)

# ann.save_model()





# rf_model = RandomForestModel()
# #old_rf_model = RandomForestModel.load_model("models/old_random_forest_model.pkl")

# # Evaluate the old model
# #old_rf_mse = old_rf_model.test(test_data[X_col], test_data[y_col])

# # Fit and evaluate the new model
# new_rf_mse = rf_model.fit_and_evaluate(training_data, test_data, X_col, y_col)

# # Save the new model
# rf_model.save_model()




# # Xgboost model
# xgboost_model = XGBoostModel.load_model(model_path="models/xgboost_model.pkl", n_estimators=100, max_depth=3, learning_rate=0.1)

# # Evaluate model performance on new data


# # Train and evaluate the model by retraining it
# mse = xgboost_model.fit_and_evaluate(training_data, test_data, X_col=X_col, y_col=y_col)

# # Save the model
# xgboost_model.save_model("models/xgboost_model.pkl")


# LSTM model
