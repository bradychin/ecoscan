#--------- 1. Import libraries ---------#
import os
import keras
import numpy as np
import pandas as pd
from keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import keras.applications.mobilenet_v2 as mobilenetv2
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


class MobileNetV2PreprocessingLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(MobileNetV2PreprocessingLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return mobilenetv2.preprocess_input(inputs)

    def get_config(self):
        # This is required to ensure proper saving/loading
        config = super(MobileNetV2PreprocessingLayer, self).get_config()
        return config

#--------- 2. Load data ---------#
def load_data(dataset_path):
    class_labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
    file_list = []
    categories = []

    for class_label in class_labels:
        filenames = os.listdir(f'{dataset_path}/{class_labels[class_label]}')
        file_dirs = [f'{class_labels[class_label]}/{filename}' for filename in filenames]
        file_list = file_list + file_dirs
        categories = categories + [class_labels[class_label]] * len(filenames)

    df = pd.DataFrame({
        'filename': file_list,
        'category': categories
    })
    df = df.sample(frac=1).reset_index(drop=True)

    # Split dataset
    train_images, validation_test_images = train_test_split(df, test_size=0.3, random_state=42)
    validation_images, test_images = train_test_split(validation_test_images, test_size=0.3, random_state=42)

    return train_images, validation_images, test_images

#--------- 3. Process Dataset ---------#
def process_data(dataset_path, train_images, validation_images, test_images, batch_size):
    # Preprocess data
    image_size = (224,224)
    class_mode = 'categorical'
    color_mode = 'rgb'

    preprocess_training = ImageDataGenerator(
        rotation_range=30,
        shear_range=0.1,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2
    )
    preprocess_validation = ImageDataGenerator()
    preprocess_testing = ImageDataGenerator()

    train_dataset = preprocess_training.flow_from_dataframe(
        train_images,
        dataset_path,
        x_col='filename',
        y_col='category',
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode=color_mode,
        shuffle=True
    )

    validation_dataset = preprocess_validation.flow_from_dataframe(
        validation_images,
        dataset_path,
        x_col='filename',
        y_col='category',
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode=color_mode,
        shuffle=False
    )

    test_dataset = preprocess_testing.flow_from_dataframe(
        test_images,
        dataset_path,
        x_col='filename',
        y_col='category',
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode=color_mode,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset

#--------- 4. Build Model ---------#
def build_model():
    mobilenetv2_layer = mobilenetv2.MobileNetV2(include_top=False,
                                                input_shape=(224, 224, 3),
                                                weights='./mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
                                                )

    # We don't want to train the imported weights
    mobilenetv2_layer.trainable = False

    model = keras.Sequential()
    model.add(keras.Input(shape=(224, 224, 3)))

    model.add(MobileNetV2PreprocessingLayer())

    model.add(mobilenetv2_layer)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(6, activation='softmax', kernel_regularizer=l2(0.001)))

    return model

#--------- 5. Train Model ---------#
def train_model(model, train_dataset, validation_dataset, batch_size):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_dataset.classes),
        y=train_dataset.classes
    )

    class_weights_dict = dict(enumerate(class_weights))

    history = model.fit(
        train_dataset,
        epochs=50,
        steps_per_epoch=len(train_dataset) // batch_size,
        validation_data=validation_dataset,
        validation_steps=len(validation_dataset) // batch_size,
        class_weight=class_weights_dict
    )

    model.save('model.keras')

    return history

#--------- 6. Evaluate / Make Predictions ---------#
def evaluate_model(model, test_dataset, history):
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Accuracy: {accuracy}. Loss: {loss}')
    # Get true labels and predictions
    y_true = test_dataset.classes
    y_pred_probs = model.predict(test_dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Create plot
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    # Plot accuracy
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[0].grid()

    # Plot loss
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Val Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid()

#--------- Main Function ---------#
def main():
    # Define original dataset
    dataset_path = 'dataset-resized'

    batch_size = 64

    train_images, validation_images, test_images = load_data(dataset_path)
    train_data, validation_data, test_data = process_data(dataset_path, train_images, validation_images, test_images, batch_size)
    model = build_model()
    history = train_model(model, train_data, validation_data, batch_size)
    evaluate_model(model, test_data, history)

if __name__ == '__main__':
    main()