from utils.load_data import get_all_files
import pandas as pd
import numpy as np
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
def save_simple_merged_data(df_lst):
    full_data = pd.concat(df_lst, ignore_index=True)
    training_data, test_data = train_test_split(full_data, random_state = 1)
    training_data.to_csv('temp/training_data_simple.csv', mode='w')
    test_data.to_csv('temp/test_data_simple.csv', mode='w')


def save_past_dependence_merged_data(df_lst, window_size: int = 10):
    full_data = pd.concat(df_lst, ignore_index=True)
    full_data.to_csv('temp/simple_milling_data.csv', mode='w')


