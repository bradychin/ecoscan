"""
Simple Model Evaluation Script
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow as tf

# Define the custom preprocessing layer
class MobileNetV2PreprocessingLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

# Load model with custom object
model = keras.models.load_model(
    './models/model.keras',
    custom_objects={'MobileNetV2PreprocessingLayer': MobileNetV2PreprocessingLayer}
)

# Load test data - UPDATE THIS PATH
test_data_dir = './data/dataset'

# Create data generator
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),  # Change if your images are different size
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Get predictions
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Calculate metrics
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average='macro')
recall = recall_score(true_classes, predicted_classes, average='macro')

# Print results
print(f"\nTest Accuracy: {accuracy * 100:.1f}%")
print(f"Precision: {precision * 100:.1f}%")
print(f"Recall: {recall * 100:.1f}%")