from utils.load_data import load_data
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

y_col = ["Dia Disp"]
X_col =  ["Ambient","Ref Temp on Bed","Spindle Rear","Coolantwall","Transfomerbed","Spindle Front","Time"]

"""
For ANN model we can merge all data in one file and result will be same
But for those model where output depends on past data then direct merging 
will lead to loss of data.
"""
# Normalization of data


# Save temp data
def load_and_concat_data():
    df_lst = load_data(3)

    full_data = pd.concat(df_lst, ignore_index=True)
    training_data, test_data = train_test_split(full_data, random_state = 1)

    os.makedirs("temp", exist_ok=True)

    training_data.to_csv('temp/training_data_simple.csv', mode='w')
    test_data.to_csv('temp/test_data_simple.csv', mode='w')


def save_past_dependence_merged_data(df_lst, window_size: int = 10):
    full_data = pd.concat(df_lst, ignore_index=True)
    full_data.to_csv('temp/simple_milling_data.csv', mode='w')


