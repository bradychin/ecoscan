# --------- Import Libraries ---------#
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from src.utils.model_net_v2_processing_layer import MobileNetV2PreprocessingLayer

# --------- Import Classes ---------#
from src.utils import config

# --------- Predictor ---------#
class Predictor:
    def predict_image(self, image_path):
        model = load_model(config.model_path, custom_objects={'MobileNetV2PreprocessingLayer': MobileNetV2PreprocessingLayer})

        image = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]  # Get the first (and only) prediction
        class_labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]
        predicted_probability = prediction[predicted_index]

        sorted_indices = np.argsort(prediction)[::-1]
        sorted_classes = [(class_labels[i], prediction[i]) for i in sorted_indices]

        print("\nPredicted class probabilities:")
        for label, prob in sorted_classes:
            print(f"{label}: {prob:.4f}")

        return predicted_class, predicted_probability, image

    def recycling_decision(self, prediction, probability):
        print(f'\nPrediction: {prediction.capitalize()}')
        print(f'Probability: {probability*100:.2f}%')
        print('\nFollow the below steps to properly recycle.')

        # Rule based decision-making
        if prediction == 'cardboard':
            print('''        1. Clean and dry.
            2. Remove packaging, tape, labels.
            3. Cut out soiled areas. 
            4. Flatten
            ''')
        elif prediction == 'glass':
            print('''        1. Empty and rinse bottles or jars.
            2. Do not remove lids.
            3. Remove corks.
            4. Ensure that glass is not broken.
            ''')
        elif prediction == 'metal':
            print('        Rules vary by location. Check you manciple recycling program.\n')
        elif prediction == 'paper':
            print('''        Ensure that paper is clean and dry. Remove staples.
            
            Paper that cannot be recycled:
            1. Coated with wax. 
            2. Lined with plastic.
            ''')
        elif prediction == 'plastic':
            print('''        Ensure that containers or bottles are empty, clean, and dry.
            
            Plastic that cannot be recycled:
            1. Plastic bags, wrap, film.
            2. Flexible packaging.
            3. Cups or containers with wax coating.
            4. Polystyrene.
            ''')
        else:
            print('You should recycle this item.\n')

# --------- Predict Image Function ---------#
def predict_image():
    image_path = input('Enter the name of your image along with the extension (example: paper.jpg)\n>>> ')

    predictor = Predictor()

    prediction, probability, image = predictor.predict_image(os.path.join(config.PROJECT_ROOT, f'./data/Test Images/{image_path}.jpg'))
    predictor.recycling_decision(prediction, probability)

    plt.imshow(image)
    plt.title(f'Prediction: {prediction}')
    plt.axis('off')
    plt.show()

# --------- Main Function ---------#
if __name__ == '__main__':
    predict_image()