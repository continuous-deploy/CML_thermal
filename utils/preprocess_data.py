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





def save_past_dependence_merged_data(window_size: int = 10):
    """
    Prepares and saves data for LSTM model with a sliding window approach.

    Parameters:
    - window_size: int, the number of past observations to consider for each sequence.
    """
    df_lst = load_data(3)  # Load list of dataframes as needed
    X_data, y_data = [], []
    save_path = 'temp/timedependent_compressed.npz'

    for df in df_lst:
        df = df.dropna().reset_index(drop=True)  # Drop NaN values and reset index if needed
        m, n = df.shape

        # Generate sequences for the current dataframe
        for i in range(m - window_size):
            # Extract window of features (excluding the first column as features)
            X = df.iloc[i:i + window_size, 1:].values  # Use all columns except the first as features
            y = df.iloc[i + window_size, 0]  # Use the target column in the first column as label

            X_data.append(X)
            y_data.append(y)

    # Convert lists to arrays
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    # Ensure directory exists and save the .npz file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, X=X_data, y=y_data)
    print(f"Data saved to {save_path}")

# Call the function
save_past_dependence_merged_data()