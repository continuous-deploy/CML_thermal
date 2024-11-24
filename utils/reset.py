import os
import shutil

# delete and create a empty 'data' folder in root to store raw data to be preprocessed
# delete and create a empty 'temp' folder in rootto store temporary preprocessed data
# delete metrics folder and create it with subfolders as ann, lstm, drift, xgboost, random_forest
# Check if excel-data file exist or not?



def reset_configuration():
    # Paths to be created or reset
    config = {
        "data_folder": "data",
        "temp_folder": "temp",
        "metrics_folder": "metrics",
        "metrics_subfolders": ["ann", "lstm", "drift", "xgboost", "random_forest", "combined_report"]
    }
    
    # Helper function to delete and recreate a folder
    def reset_folder(folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
    
    # Reset 'data' folder
    reset_folder(config['data_folder'])
    
    # Reset 'temp' folder
    reset_folder(config['temp_folder'])
    
    # Reset 'metrics' folder with subfolders
    reset_folder(config['metrics_folder'])

    for subfolder in config['metrics_subfolders']:
        os.makedirs(os.path.join(config['metrics_folder'], subfolder))


if __name__ == "__main__":
    reset_configuration()