import os

data_folder_path = "data"

def get_all_files(data_folder_path:str = data_folder_path):
    file_lst = []
    for root, _, files in os.walk(data_folder_path):
        for file in files:
            file_lst.append(os.path.join(root, file))
    
    return file_lst