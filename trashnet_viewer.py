# Import libraries
import os
import matplotlib.pyplot as plt
from PIL import Image
import random
import pandas as pd

# Verify dataset path
dataset_path = './dataset-resized'
categories = os.listdir(dataset_path)

print(f"Categories: {categories}")

# Preview random image from dataset
category = random.choice(categories)
category_path = os.path.join(dataset_path, category)
sample_image = random.choice(os.listdir(category_path))

img_path = os.path.join(category_path, sample_image)
img = Image.open(img_path)

plt.imshow(img)
plt.title(f'Category: {category}')
plt.axis('off')

# Convert folder structure to CSV
data = []
for category in categories:
    category_path = os.path.join(dataset_path, category)
    for img in os.listdir(category_path):
        data.append((os.path.join(category_path, img), category))

df = pd.DataFrame(data, columns=['image_path', 'label'])
print(df.head())
df.to_csv('trashnet_dataset.csv', index=False)
print('File saved as trashnet_dataset.csv')

# Class distribution
df = pd.read_csv('trashnet_dataset.csv')
class_counts = df['label'].value_counts().reset_index() # count number of images per class
class_counts.columns = ['Class', 'Number of Images']
print(class_counts)

# visualize
# plt.figure(figsize=(8,5))
class_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.show()