from utils.load_data import get_all_files
import pandas as pd
import numpy as np

file_lst  = get_all_files()

df_lst = [pd.read_csv(file_) for file_ in file_lst]

# Dia Disp,Ambient,Ref Temp on Bed,Spindle Rear,Coolantwall,Transfomerbed,Spindle Front,Time
data_col = df_lst[0].columns

y_col = data_col[0]
X_col = data_col[1:]

"""
For ANN model we can merge all data in one file and result will be sanme
But for those model where output depends on past data then direct merging 
will lead to loss of data.

"""
# Normalization of data



# Save temp data
def save_simple_merged_data(df_lst):
    full_data = pd.concat(df_lst, ignore_index=True)
    full_data.to_hdf('Temp/simple_milling_data.h5', key='data', mode='w')


def save_past_dependence_merged_data(df_lst, window_size: int = 10):
    full_data = pd.concat(df_lst, ignore_index=True)
    full_data.to_hdf('Temp/simple_milling_data.h5', key='data', mode='w')

