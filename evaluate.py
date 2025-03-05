from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array
import random
import keras
from ecoscan import MobileNetV2PreprocessingLayer

model = load_model('model.keras', custom_objects={'MobileNetV2PreprocessingLayer': MobileNetV2PreprocessingLayer})



class_id = 'paper'
predict_image_path = f'./dataset-resized/{class_id}/{class_id}{random.randint(1, 300)}.jpg'
img = load_img(predict_image_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)[0]  # Get the first (and only) prediction
class_labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
predicted_index = np.argmax(prediction)
predicted_class = class_labels[predicted_index]

print(f'\nActual class: {class_id}')
print(f'Predicted class: {predicted_class}')

sorted_indices = np.argsort(prediction)[::-1]
sorted_classes = [(class_labels[i], prediction[i]) for i in sorted_indices]
print("Predicted class probabilities:")
for label, prob in sorted_classes:
    print(f"{label}: {prob:.4f}")

plt.imshow(img)
plt.title(f'Predicted: {predicted_class}')
plt.axis('off')
plt.show()

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

    print('\nTraining history:')
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    df = pd.DataFrame(history.history)
    print(df)

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