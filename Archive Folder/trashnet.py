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

# Set the directory path
dir_path = '../dataset-resized'
class_labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
# Create an empty list to store the file names and paths
file_list = []
categories = []

# Walk through the directory and its subdirectories
for class_label in class_labels:
    filenames = os.listdir(f'{dir_path}/{class_labels[class_label]}')
    file_dirs = [f'{class_labels[class_label]}/{filename}' for filename in filenames]
    file_list = file_list + file_dirs
    categories = categories + [class_labels[class_label]] * len(filenames)

# Create a Pandas DataFrame from the list
df = pd.DataFrame({
    'filename': file_list,
    'category': categories
})
df = df.sample(frac=1).reset_index(drop=True)
print(df)