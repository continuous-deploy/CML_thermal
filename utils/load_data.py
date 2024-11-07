import os
"""
From here we will make sure that we are consiidering data of past 3 sessions(day/ file/ etc)


"""
data_folder_path = "data"

def get_all_files(data_folder_path:str = data_folder_path):
    file_lst = []
    for root, _, files in os.walk(data_folder_path):
        for file in files:
            file_lst.append(os.path.join(root, file))
    print(file_lst)
    return file_lst
