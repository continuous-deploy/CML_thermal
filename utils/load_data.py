import os
import pandas as pd
from typing import List
from pandas import DataFrame

data_folder_path = "data"

def get_most_recent_files(data_folder_path: str = data_folder_path, num_files: int = 3) -> List[str]:
    file_lst = []
    
    # Get all files in the directory
    for root, _, files in os.walk(data_folder_path):
        for file in files:
            if file.endswith('.csv'):  # Only consider CSV files
                file_path = os.path.join(root, file)
                # Append file path and its modification time
                file_lst.append((file_path, os.path.getmtime(file_path)))

    # Sort files by modification time in descending order and get the most recent ones
    most_recent_files = sorted(file_lst, key=lambda x: x[1], reverse=False)[-num_files:]
    
    # Extract just the file paths
    most_recent_files = [file[0] for file in most_recent_files]
    
    return most_recent_files

def load_data(file_count: int = 3) -> List[DataFrame]:
    file_lst = get_most_recent_files(data_folder_path, file_count)
    df_lst = [pd.read_csv(file_) for file_ in file_lst]
    return df_lst
