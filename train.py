"""
This file will trigger model retrainning after importingg all the models from model folder

As this file is triggered, It should do folowing task:
1. 
"""

import pandas as pd

from models.ann import ANN
from models.lstm import LSTM

from utils.preprocess_data import load_and_concat_data
from utils.preprocess_data import X_col, y_col


load_and_concat_data()

training_data = pd.read_csv('temp/training_data_simple.csv')
test_data = pd.read_csv('temp/test_data_simple.csv')


ann = ANN()

hist = ann.fit(training_data[X_col], training_data[y_col])

ann.test(test_data[X_col], test_data[y_col])

ann.save_model()

#Save into metrics

