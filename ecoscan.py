#--------- 1. Import libraries ---------#
import os
import random
import shutil

from sklearn.model_selection import train_test_split

#--------- 2. Process Dataset ---------#
dataset_path = './dataset-resized'

# Create new subdirectories
organized_path = './dataset-organized'
os.makedirs(os.path.join(organized_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(organized_path, 'validation'), exist_ok=True)
os.makedirs(os.path.join(organized_path, 'test'), exist_ok=True)

classes = [cls for cls in os.listdir(dataset_path) if cls != ".DS_Store"]
print(classes)
for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)

    # Create directories for train, test, and validation sets for each class
    os.makedirs(os.path.join(organized_path, 'train', class_name), exist_ok=True)
    os.makedirs(os.path.join(organized_path, 'test', class_name), exist_ok=True)
    os.makedirs(os.path.join(organized_path, 'validation', class_name), exist_ok=True)

    # Split the dataset into train, test, and validation sets for each class
    train_images, test_val_images = train_test_split(os.listdir(class_path), test_size=0.3, random_state=42)
    test_images, validation_images = train_test_split(test_val_images, test_size=0.5, random_state=42)
    # Copy images to train, test, and validation folders for each class
    for image in train_images:
        shutil.copy(os.path.join(class_path, image), os.path.join(organized_path, 'train', class_name, image))
    for image in test_images:
        shutil.copy(os.path.join(class_path, image), os.path.join(organized_path, 'test', class_name, image))
    for image in validation_images:
        shutil.copy(os.path.join(class_path, image), os.path.join(organized_path, 'validation', class_name, image))

#--------- 3. Build Model ---------#


#--------- 4. Train Model ---------#





