#--------- 1. Import libraries ---------#
import os
import pandas as pd

#--------- 2. Load Dataset ---------#
dataset_path = './dataset-resized'
categories = os.listdir(dataset_path)
print(f'Categories: {categories}')

# Convert folder structure to CSV
data = []
for category in categories:
    category_path = os.path.join(dataset_path, category)
    for img in os.listdir(category_path):
        data.append((os.path.join(category_path, img), category))
df = pd.DataFrame(data, columns=['image_path', 'label'])
df.to_csv('trashnet_dataset.csv', index=False)
print('File saved as trashnet_dataset.csv')

#--------- 3. Extract Features and Target ---------#
#--------- 4. Split Data ---------#
#--------- 5. Train the Model ---------#
#--------- 6. Make Prediction ---------#
#--------- 7. Evaluate Model ---------#