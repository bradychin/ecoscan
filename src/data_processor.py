# --------- Import Libraries ---------#
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --------- Import Classes ---------#
from src.utils import config

# --------- Data Processor ---------#
class DataProcessor:
    def load_data(self):
        class_labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
        file_list = []
        categories = []

        for class_label in class_labels:
            filenames = os.listdir(f'{config.dataset_path}/{class_labels[class_label]}')
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

    def process_data(self, train_images, validation_images, test_images):
        # Preprocess data
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
            config.dataset_path,
            x_col='filename',
            y_col='category',
            target_size=config.image_size,
            batch_size=config.batch_size,
            class_mode=config.class_mode,
            color_mode=config.color_mode,
            shuffle=True
        )

        validation_dataset = preprocess_validation.flow_from_dataframe(
            validation_images,
            config.dataset_path,
            x_col='filename',
            y_col='category',
            target_size=config.image_size,
            batch_size=config.batch_size,
            class_mode=config.class_mode,
            color_mode=config.color_mode,
            shuffle=False
        )

        test_dataset = preprocess_testing.flow_from_dataframe(
            test_images,
            config.dataset_path,
            x_col='filename',
            y_col='category',
            target_size=config.image_size,
            batch_size=config.batch_size,
            class_mode=config.class_mode,
            color_mode=config.color_mode,
            shuffle=False
        )

        return train_dataset, validation_dataset, test_dataset

def data_processor():
    data = DataProcessor()
    train_images, validation_images, test_images = data.load_data()
    train_data, validation_data, test_data = data.process_data(train_images,
                                                               validation_images,
                                                               test_images)

    return train_data, validation_data, test_data