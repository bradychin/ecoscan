#--------- 1. Import libraries ---------#
import tensorflow as tf
import random
from tf_keras.src.preprocessing.image import ImageDataGenerator

#--------- 2. Load Dataset ---------#
dataset_path = '../dataset-resized'

# Resize images to 224x224 px for CNN compatibility
resized_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(224,224),
    batch_size=32,
    shuffle=True
)

# Normalize data
normalization_layer = tf.keras.layers.Rescaling(1./255)
resized_dataset = resized_dataset.map(lambda x, y: (normalization_layer(x), y))

# Data augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# img = load_img('../dataset-resized/cardboard/cardboard1.jpg')  # this is a PIL image
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#
# # the .flow() command below generates batches of randomly transformed images
# # and saves the results to the `preview/` directory
# i = 0
# for batch in data_augmentation.flow(x, batch_size=1,
#                           save_to_dir='test', save_prefix='test', save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely
#
# import numpy as np
#
# # Create dummy data: 100 items and 3 classes (0, 1, or 2)
# dummy_data = np.random.randint(0, 3, size=(100,))
# print(dummy_data)
# print('\n')
# # Shuffle and split data
# np.random.shuffle(dummy_data)
# print(dummy_data)
# print('\n')
# train_size = int(0.8 * len(dummy_data))
# print(f'train_size={train_size}')
# validation_size = int(0.1 * len(dummy_data))
# print(f'train_size={validation_size}')
# test_size = int(0.1 * len(dummy_data))
# print(f'train_size={test_size}')
# print('\n')
#
# # Split into train, validation, and test sets
# train_data = dummy_data[:train_size]
# print(f'train_data={train_data}')
# validation_data = dummy_data[train_size:train_size + validation_size]
# print(f'validation_data={validation_data}')
# test_data = dummy_data[train_size + validation_size:]
# print(f'test_data={test_data}')
# print('\n')
#
# # Check the sizes of the resulting datasets
# print(f"Train data size: {len(train_data)}")
# print(f"Validation data size: {len(validation_data)}")
# print(f"Test data size: {len(test_data)}")
#
# # Data augmentation (prevent overfitting)
# def data_augmentation(image, label):
#     image = tf.image.rot90(image)
#     image = tf.image.flip_left_right(image)
#     image = tf.image.flip_up_down(image)
#     image = tf.image.adjust_brightness(image, delta=random.uniform(0.0, 1.0))
#     return image, label
#
# # Apply augmentation only to training dataset
# train_dataset = train_dataset.map(data_augmentation)
#
# import matplotlib.pyplot as plt
#
# # Take a batch from train dataset
# for images, labels in train_dataset.take(1):
#     fig, axes = plt.subplots(1, 5, figsize=(15, 5))
#
#     for i in range(5):
#         axes[i].imshow(images[i].numpy())
#         axes[i].axis("off")
#
#     plt.show()