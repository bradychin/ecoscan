# --------- Import Libraries ---------#
import os

# --------- Import Classes ---------#
from src.model_wrapper import model_wrapper
from src.predictor import predict_image
from src.utils.config import model_path

# --------- Main Function ---------#
def main():
    if not os.path.exists(model_path):
        print('No model. Starting training.')
        model_wrapper()

    print('Predict')
    predict_image()

if __name__ == '__main__':
    main()