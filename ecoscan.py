#--------- 1. Import libraries ---------#
import os
import shutil
import keras
import numpy as np
from keras import layers
from keras.optimizers import Adam
from keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#--------- 2. Load data ---------#
def load_data(dataset_path, organized_path):
    os.makedirs(os.path.join(organized_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(organized_path, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(organized_path, 'test'), exist_ok=True)

    classes = [cls for cls in os.listdir(dataset_path) if cls != ".DS_Store"]
    print(classes)

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)

        # Create train, validation, test directories in organized dataset
        os.makedirs(os.path.join(organized_path, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(organized_path, 'test', class_name), exist_ok=True)
        os.makedirs(os.path.join(organized_path, 'validation', class_name), exist_ok=True)

        # Split dataset
        train_images, test_val_images = train_test_split(os.listdir(class_path), test_size=0.3, random_state=42)
        test_images, validation_images = train_test_split(test_val_images, test_size=0.5, random_state=42)

        # Copy images into organized folders
        for image in train_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(organized_path, 'train', class_name, image))
        for image in test_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(organized_path, 'test', class_name, image))
        for image in validation_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(organized_path, 'validation', class_name, image))

#--------- 3. Process Dataset ---------#
def process_data(organized_path):
    # Preprocess data
    rescale = 1./255
    image_size = (224,224)
    batch_size = 32
    class_mode = 'sparse'
    color_mode = 'rgb'

    preprocess_training = ImageDataGenerator(
        rescale=rescale,
        rotation_range=20,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip = True,
        vertical_flip = True
    )
    preprocess_validation = ImageDataGenerator(rescale=rescale)
    preprocess_testing = ImageDataGenerator(rescale=rescale)

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

    return train_dataset, validation_dataset, test_dataset

#--------- 4. Build Model ---------#
def build_model():
    model = keras.Sequential()

    model.add(keras.Input(shape=(224,224,3))) # Input layer

    # Hidden layers (5)
    model.add(layers.Conv2D(64, (3,3))) # Convolution layer
    model.add(layers.Activation('relu')) # Activation Layer
    model.add(layers.MaxPool2D(pool_size=(2, 2)))  # Pooling layer
    model.add(layers.Conv2D(128, (3,3))) # Convolution layer
    model.add(layers.Activation('relu')) # Activation Layer
    model.add(layers.MaxPool2D(pool_size=(2,2))) # Pooling layer
    model.add(layers.Conv2D(156, (3, 3)))  # 3rd convolutional layer
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))  # Pooling after each convolutional block

    model.add(layers.Flatten()) # Flatten layer
    model.add(layers.Dense(128, activation='relu'))# Fully Connected (Dense) layer
    model.add(layers.Dense(6, activation='softmax'))# Output layer

    return model

#--------- 5. Train Model ---------#
def train_model(model, train_dataset, validation_dataset):
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.fit(train_dataset,
              steps_per_epoch=8,
              epochs=50,
              validation_data=validation_dataset,
              validation_steps=8)

#--------- 6. Evaluate / Make Predictions ---------#
def evaluate_mode(model, test_dataset):
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Accuracy: {accuracy}. Loss: {loss}')

def single_image_prediction(model):
    predict_image_path = './dataset-organized/test/metal/metal29.jpg'
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

#--------- Main Function ---------#
def main():
    # Define original dataset
    dataset_path = './dataset-resized'
    # Create new directory
    organized_path = './dataset-organized'

    load_data(dataset_path, organized_path)
    train_data, validation_data, test_data = process_data(organized_path)
    model = build_model()
    train_model(model, train_data, validation_data)
    evaluate_mode(model, test_data)
    single_image_prediction(model)

if __name__ == '__main__':
    main()