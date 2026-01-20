# Waste Classification Model

A machine learning project that uses transfer learning with MobileNetV2 to classify waste images into different categories for recycling guidance.

## Requirements
- Python 3.11 or 3.12

## Setup

1. Clone this repository:
```bash
git clone https://github.com/bradychin/ecoscan
cd ecoscan
```

2. Create a virtual environment with Python 3.11:

On macOS
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```
On Windows
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Usage

### Quick Start - Test Predictions
The project comes with a pre-trained model and sample test images. Simply run:
```bash
python main.py
```

When prompted, enter the name of an image file from the `test images/` folder (e.g., `sample.jpg`).

You can add your own test images to the `test images/` folder - supports `.jpg`, `.jpeg`, and `.png` formats.

### Retrain the Model
To retrain with your own data:

1. Organize your training images in `data/dataset/` by category:
```
data/dataset/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

2. Delete or rename the existing model:
```bash
rm models/model.keras
```

3. Run the training:
```bash
python main.py
```

### Evaluate Model Performance
To see accuracy, precision, and recall metrics:
```bash
python evaluation.py
```

## Project Structure
```
.
├── main.py              # Main entry point
├── evaluation.py        # Model evaluation script
├── src/                 # Source code
│   ├── model_wrapper.py
│   ├── predictor.py
│   └── utils/
│       └── config.py
├── data/                # Training data (provided)
│   └── dataset/
├── models/              # Pre-trained model (included)
│   └── model.keras
├── test images/         # Sample test images (provided)
└── requirements.txt     # Python dependencies
```

## Model Details
- Base Model: MobileNetV2 (pretrained on ImageNet)
- Transfer Learning: Fine-tuned on waste classification dataset
- Image Size: 224x224
- Classes: Cardboard, Glass, Metal, Paper, Plastic, Trash
- Evaluation Metrics: Accuracy, Precision, Recall

## Customization
- Replace images in `data/dataset/` with your own training data
- Add your own test images to `test images/`
- Modify class names in the code to match your categories
