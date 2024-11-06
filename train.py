"""
This file will trigger model retrainning after importingg all the models from model folder
"""

import pandas as pd
from sklearn.model_selection import train_test_split 

from models.ann import ANN
from utils.load_data import get_all_files
from utils.preprocess_data import save_simple_merged_data
from utils.preprocess_data import X_col, y_col


file_lst = get_all_files("data")
df_lst = [pd.read_csv(file_) for file_ in file_lst]
save_simple_merged_data(df_lst)

df = pd.read_csv('temp/simple_milling_data.csv')


ann = ANN()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(df[X_col],df[y_col])

hist = ann.fit(Xtrain, Ytrain)

ann.test(Xtest, Ytest)



#Save into metrics
