"""
This file will trigger model retrainning after importingg all the models from model folder

As this file is triggered, It should do folowing task:
1. 
"""
import pandas as pd
from sklearn.model_selection import train_test_split 
from models.ann import ANN
from models.lstm import LSTM
from utils.load_data import get_all_files
from utils.preprocess_data import save_simple_merged_data
from utils.preprocess_data import X_col, y_col


file_lst = get_all_files("data")

df_lst = [pd.read_csv(file_) for file_ in file_lst]

save_simple_merged_data(df_lst)

training_data = pd.read_csv('temp/training_data_simple.csv')
test_data = pd.read_csv('temp/test_data_simple.csv')



ann = ANN()

hist = ann.fit(training_data[X_col], training_data[y_col])

ann.test(test_data[X_col], test_data[y_col])

ann.save_model()

#Save into metrics

