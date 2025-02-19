#--------- 1. Import libraries ---------#
import keras
import tensorflow as tf
import random
from keras import layers

#--------- 2. Process Dataset ---------#
dataset_path = './dataset-resized'

# Resize images to 224x224 px for CNN compatibility
resized_dataset = keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(224,224),
    batch_size=32,
    shuffle=True
)

# Normalize data
normalization_layer = keras.layers.Rescaling(1./255)
resized_dataset = resized_dataset.map(lambda x, y: (normalization_layer(x), y))

# Split data
resized_dataset = resized_dataset.shuffle(buffer_size=1000)
train_size = int(0.7 * len(resized_dataset))
validation_size = int(0.15 * len(resized_dataset))
test_size = int(0.15 * len(resized_dataset))

train_dataset = resized_dataset.take(train_size)
validation_dataset = resized_dataset.skip(train_size).take(validation_size)
test_dataset = resized_dataset.skip(train_size+validation_size).take(test_size)

# Data augmentation (prevent overfitting)
def data_augmentation(image, label):
    image = tf.image.rot90(image)
    image = tf.image.flip_left_right(image)
    image = tf.image.flip_up_down(image)
    image = tf.image.adjust_brightness(image, delta=random.uniform(0.0, 1.0))
    return image, label

# Apply augmentation only to training dataset
train_dataset = train_dataset.map(data_augmentation)

#--------- 3. Build Model ---------#
model = keras.Sequential()

model.add(keras.Input(shape=(224,224,3))) # Input layer

# Hidden layers (5)
model.add(layers.Conv2D(32, (3,3))) # Convolution layer
model.add(layers.Activation('relu')) # Activation Layer
model.add(layers.Conv2D(64, (3,3))) # Convolution layer
model.add(layers.Activation('relu')) # Activation Layer
model.add(layers.MaxPool2D(pool_size=(2,2))) # Pooling layer

model.add(layers.Flatten()) # Flatten layer
model.add(layers.Dense(128, activation='relu'))# Fully Connected (Dense) layer
model.add(layers.Dense(6, activation='softmax'))# Output layer

#--------- 4. Train Model ---------#











