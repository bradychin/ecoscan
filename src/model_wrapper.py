# --------- Import Libraries ---------#
from keras import Sequential, Input, layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
import keras.applications.mobilenet_v2 as mobilenetv2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# --------- Import Classes ---------#
from src.utils.model_net_v2_processing_layer import MobileNetV2PreprocessingLayer
from src.data_processor import data_processor
from src.utils import config

# --------- Model Wrapper ---------#
class ModelWrapper:
    def build_model(self):
        base_model = mobilenetv2.MobileNetV2(include_top=False,
                                             input_shape=(*config.image_size, 3),
                                             weights= config.model_weights)

        # We don't want to train the imported weights
        base_model.trainable = False

        model = Sequential([Input(shape=(*config.image_size, 3)),
                            MobileNetV2PreprocessingLayer(),
                            base_model,
                            layers.GlobalAveragePooling2D(),
                            layers.Dense(config.num_classes, activation='softmax', kernel_regularizer=l2(0.001))])

        return model

    def train(self, model, train_dataset, validation_dataset):
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_dataset.classes),
            y=train_dataset.classes)

        class_weights_dict = dict(enumerate(class_weights))

        history = model.fit(train_dataset,
                            epochs=config.epochs,
                            steps_per_epoch=len(train_dataset) // config.batch_size,
                            validation_data=validation_dataset,
                            validation_steps=len(validation_dataset) // config.batch_size,
                            class_weight=class_weights_dict)

        model.save('model.keras')

        return history

    def evaluate_model(self, model, test_dataset, history):
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

# --------- Model Wrapper Function ---------#
def model_wrapper():
    train_data, validation_data, test_data = data_processor()

    wrapper = ModelWrapper()
    model = wrapper.build_model()
    history = wrapper.train(model, train_data, validation_data)
    wrapper.evaluate_model(model, test_data, history)

# --------- Main Function ---------#
if __name__ == '__main__':
    model_wrapper()