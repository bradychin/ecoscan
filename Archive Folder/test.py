import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import sys
import time
import tensorflow as tf
import re

from PIL import Image
import keras
from keras import layers
from keras.layers import Lambda
import keras.applications.mobilenet_v2 as mobilenetv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print('setup successful!')

# Increasing the image size didn't result in increasing the training accuracy
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3


# Path where our data is located
base_path = "../dataset-resized/"


# Dictionary to save our 12 classes
categories = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

print('defining constants successful!')


# Add class name prefix to filename. So for example "/paper104.jpg" become "paper/paper104.jpg"
def add_class_name_prefix(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: x[:re.search("\d", x).start()] + '/' + x)
    return df


# list conatining all the filenames in the dataset
filenames_list = []
# list to store the corresponding category, note that each folder of the dataset has one class of data
categories_list = []

for category in categories:
    filenames = os.listdir(base_path + categories[category])

    filenames_list = filenames_list + filenames
    categories_list = categories_list + [category] * len(filenames)

df = pd.DataFrame({
    'filename': filenames_list,
    'category': categories_list
})

df = add_class_name_prefix(df, 'filename')

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

print('number of elements = ', len(df))

print(df.head())
#
# mobilenetv2_layer = mobilenetv2.MobileNetV2(include_top = False,
#                                             input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS),
#                                             weights = '../mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')
#
# # We don't want to train the imported weights
# mobilenetv2_layer.trainable = False
#
#
# model = keras.Sequential()
# model.add(keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
#
# #create a custom layer to apply the preprocessing
# def mobilenetv2_preprocessing(img):
#   return mobilenetv2.preprocess_input(img)
#
# model.add(Lambda(mobilenetv2_preprocessing))
#
# model.add(mobilenetv2_layer)
# model.add(layers.GlobalAveragePooling2D())
# model.add(layers.Dense(len(categories), activation='softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.summary()
#
# early_stop = EarlyStopping(patience = 2,
#                            verbose = 1,
#                            monitor='val_categorical_accuracy',
#                            mode='max',
#                            min_delta=0.001,
#                            restore_best_weights = True)
#
# callbacks = [early_stop]
#
# print('call back defined!')
#
# #Change the categories from numbers to names
# df["category"] = df["category"].replace(categories)
#
# # We first split the data into two sets and then split the validate_df to two sets
# train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
# validate_df, test_df = train_test_split(validate_df, test_size=0.3, random_state=42)
#
# train_df = train_df.reset_index(drop=True)
# validate_df = validate_df.reset_index(drop=True)
# test_df = test_df.reset_index(drop=True)
#
# total_train = train_df.shape[0]
# total_validate = validate_df.shape[0]
#
# print('train size = ', total_validate , 'validate size = ', total_validate, 'test size = ', test_df.shape[0])
#
# batch_size = 64
#
# train_datagen = ImageDataGenerator(
#
#     ###  Augmentation Start  ###
#
#     rotation_range=30,
#     shear_range=0.1,
#     zoom_range=0.3,
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.2,
#     height_shift_range=0.2
#
#     ##  Augmentation End  ###
# )
#
# train_generator = train_datagen.flow_from_dataframe(
#     train_df,
#     base_path,
#     x_col='filename',
#     y_col='category',
#     target_size=IMAGE_SIZE,
#     class_mode='categorical',
#     batch_size=batch_size
# )
#
# validation_datagen = ImageDataGenerator()
#
# validation_generator = validation_datagen.flow_from_dataframe(
#     validate_df,
#     base_path,
#     x_col='filename',
#     y_col='category',
#     target_size=IMAGE_SIZE,
#     class_mode='categorical',
#     batch_size=batch_size
# )
# EPOCHS = 50
# history = model.fit(
#     train_generator,
#     epochs=EPOCHS,
#     validation_data=validation_generator,
#     validation_steps=total_validate//batch_size,
#     steps_per_epoch=total_train//batch_size,
#     #callbacks=callbacks
# )