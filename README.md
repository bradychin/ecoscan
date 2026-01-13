# Waste Classification Model

A machine learning project that uses transfer learning with MobileNetV2 to classify waste images into different categories for recycling guidance.

## Requirements
- Python 3.11 or 3.12

## Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Create a virtual environment with Python 3.11:
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create the required directories:
```bash
mkdir -p data/dataset models
```

5. Add your training data to `data/dataset/` organized by class folders (e.g., `data/dataset/cardboard/`, `data/dataset/plastic/`, etc.)

## Usage

1. Place your image into the "test images" folder
2. run the main script
```bash
python main.py
```

This will:
- Train a model if one doesn't exist yet (saves to `models/model.keras`)
- Run predictions on images

To evaluate the model:
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
├── data/                # Training data
│   └── dataset/
├── models/              # Saved models
└── requirements.txt     # Python dependencies
```

## Model Details
- Base Model: MobileNetV2 (pretrained on ImageNet)
- Image Size: 224x224
- Evaluation Metrics: Accuracy, Precision, Recall