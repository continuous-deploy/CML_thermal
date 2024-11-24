import os
import pandas as pd
import re
import shutil
from train import analyse_model_and_make_report
from utils.reset import reset_configuration
from utils.check_drift import evaluate_drift, visualize_drift
from utils.segregated_excel import extract_sheets_from_excel

# Reset file data file and folders
reset_configuration()

# Extract sheets from excel file as csv
extract_sheets_from_excel()

# To disable floating-point round-off errors from different computation orders
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Source and destination directories
source_dir = 'excel_csv'
destination_dir = 'data'  # Directory where files should be moved

# Step 1: Get all CSV files from the source directory
csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]

# Extracting number from file name to sort them in numeric order
def extract_number_from_filename(filename):
    # Regular expression to extract the number after 'TM'
    match = re.search(r'TM(\d+)', filename)
    if match:
        return int(match.group(1))  # Return the number as an integer
    return float('inf')  # Return a high value if no number is found (to handle invalid filenames)

# Sort the list of files numerically based on the extracted number
csv_files.sort(key=extract_number_from_filename)

# Step 2: Make a list of full file paths
file_paths = [os.path.join(source_dir, file) for file in csv_files]


ref_file = pd.read_csv(file_paths[0])

# Step 3: Iterate through the list and move each file
for file_path in file_paths:
    # Construct the destination file path
    destination_path = os.path.join(destination_dir, os.path.basename(file_path))
    
    curr_file = pd.read_csv(destination_path)


    # Move the file to the destination folder
    shutil.move(file_path, destination_path)
    print(f"{destination_path}  &&&&  {file_path}")
    
    # Step 4: Call the function to execute further steps on the moved file
    analyse_model_and_make_report()

