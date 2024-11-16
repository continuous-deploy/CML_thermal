import pandas as pd
import re
import os


# Load the workbook

file_path = 'excel_csv/ACE_data.xlsx'

if not os.path.exists(file_path):
    os.mkdir('excel_csv')

excel = pd.ExcelFile(file_path)

# Regular expression to match sheet names ending with 'TM' followed by a number
pattern = r'.* (TM\d+)$'

# Loop through each sheet and check if the name matches the pattern
for sheet_name in excel.sheet_names:
    match = re.search(pattern, sheet_name)  # Check if the sheet name matches the pattern
    if match:
        # Extract the TM# part from the sheet name
        export_name = match.group(1)
        
        # Read the matching sheet into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Save the DataFrame as a CSV file with the extracted name
        csv_file_name = f'excel_csv/{export_name}.csv'
        df.to_csv(csv_file_name, index=False)
        print(f"Exported {sheet_name} to {csv_file_name}")
