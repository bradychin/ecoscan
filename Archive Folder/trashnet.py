# #--------- 1. Import libraries ---------#
# import os
# import pandas as pd
#
# #--------- 2. Load Dataset ---------#
# dataset_path = '../dataset-resized'
# class_labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
#
# for directory_path, directory_name, filename in os.walk(dataset_path):
#     print(f"Directory: {directory_path}")
#     print(f"Subdirectories: {directory_name}")
#     print(f"Files: {filename}")
#     print("-" * 20)
#

"""
create a table containing the names of the files and paths of all the files in the shared folder
"""
import os
import pandas as pd
print(os.listdir('../dataset-resized'))
# # Set the directory path
# dir_path = '../dataset-organized'
# class_labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
# # Create an empty list to store the file names and paths
# file_list = []
#
# # Walk through the directory and its subdirectories
# for root, dirs, files in os.walk(dir_path):
#     for file in files:
#         # Get the file path
#         file_path = os.path.join(root, file)
#         # Append the file name and path to the list
#         file_list.append({'File Name': file, 'File Path': file_path})
#
# print(file_list)
#
# # Create a Pandas DataFrame from the list
# df_path = pd.DataFrame([file_list, class_labels])
#
# print(df_path.head())