#--------- 1. Import libraries ---------#
import os
import shutil
import keras
import numpy as np
from keras import layers
from keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#--------- 2. Load data ---------#
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

#--------- 3. Process Dataset ---------#
# Preprocess data
rescale = 1./255
preprocess_training = ImageDataGenerator(
    rescale=rescale,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2
)
preprocess_validation = ImageDataGenerator(rescale=rescale)
preprocess_testing = ImageDataGenerator(rescale=rescale)

# Load data
image_size = (224,224)
batch_size = 32
class_mode = 'sparse'
color_mode = 'rgb'

train_dataset = preprocess_training.flow_from_directory(
    os.path.join(organized_path, 'train'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode=class_mode,
    color_mode=color_mode,
    shuffle=True
)

validation_dataset = preprocess_validation.flow_from_directory(
    os.path.join(organized_path, 'validation'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode=class_mode,
    color_mode=color_mode,
    shuffle=False
)

test_dataset = preprocess_testing.flow_from_directory(
    os.path.join(organized_path, 'test'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode=class_mode,
    color_mode=color_mode,
    shuffle=False
)

#--------- 4. Build Model ---------#
model = keras.Sequential()

model.add(keras.Input(shape=(224,224,3))) # Input layer

# Hidden layers (5)
model.add(layers.Conv2D(32, (3,3))) # Convolution layer
model.add(layers.Activation('relu')) # Activation Layer
model.add(layers.Conv2D(64, (3,3))) # Convolution layer
model.add(layers.Activation('relu')) # Activation Layer
model.add(layers.MaxPool2D(pool_size=(2,2))) # Pooling layer

model.add(layers.Flatten()) # Flatten layer
model.add(layers.Dense(64, activation='relu'))# Fully Connected (Dense) layer
model.add(layers.Dense(6, activation='softmax'))# Output layer

#--------- 5. Train Model ---------#
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy,
              metrics=['accuracy'])

model.fit(train_dataset,
          steps_per_epoch=2,
          epochs=50,
          validation_data=validation_dataset,
          validation_steps=2)

#--------- 6. Make Predictions ---------#
predict_image_path = './dataset-organized/test/cardboard/cardboard3.jpg'
img = load_img(predict_image_path, target_size=(224,224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
print(f'The predicted probability is: {prediction}')

class_indices = np.argmax(prediction, axis=1)
class_labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
predicted_class = class_labels[class_indices[0]]

print(f'Predicted class: {predicted_class}')
plt.imshow(img)
plt.show()
