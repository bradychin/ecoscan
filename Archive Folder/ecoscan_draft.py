#--------- 1. Import libraries ---------#
import keras
import tensorflow as tf
from keras import layers

#--------- 2. Process Dataset ---------#
dataset_path = '../dataset-resized'

# Resize images to 224x224 px for CNN compatibility
resized_dataset = keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(224,224),
    batch_size=64,
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
test_dataset = resized_dataset.skip(train_size+validation_size)

# Data augmentation (prevent overfitting)
def data_augmentation(image, label):
    image = tf.image.rot90(image)
    image = tf.image.flip_left_right(image)
    image = tf.image.flip_up_down(image)
    # image = tf.image.adjust_brightness(image, delta=random.uniform(0.0, 1.0))
    return image, label

# Apply augmentation only to training dataset
train_dataset = train_dataset.map(data_augmentation)

# train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

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
model.add(layers.Dense(64, activation='relu'))# Fully Connected (Dense) layer
model.add(layers.Dense(6, activation='softmax'))# Output layer

#--------- 4. Train Model ---------#
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset,
          epochs=50,
          validation_data=validation_dataset)

y_pred = model.predict(test_dataset)
print(y_pred)

import matplotlib.pyplot as plt
import numpy as np

# Take a batch from the test dataset
for images, labels in test_dataset.take(1):
    # Get the predicted probabilities for the batch
    y_pred = model.predict(images)

    # Get the predicted class for each image
    predicted_classes = np.argmax(y_pred, axis=1)

    # Plot the first 5 images with their corresponding predicted labels
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    for i in range(5):
        axes[i].imshow(images[i].numpy())  # Display image
        axes[i].axis("off")
        axes[i].set_title(f"Pred: {predicted_classes[i]}")  # Display prediction

    plt.show()



