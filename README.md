# Computer Vision Project

This project implements a simple computer vision pipeline for image processing and analysis. It leverages OpenCV for handling image operations and Matplotlib for data visualization. The project focuses on image transformations, object detection, and result plotting.

## Features
- Load, process, and display images.
- Apply basic image transformations (grayscale, blur, edge detection, etc.).
- Visualize image data and processing results.

## Technologies Used
- **Python**: Main programming language.
- **OpenCV**: Image processing and computer vision operations.
- **Matplotlib**: Visualization of images and processing results.

## Project Overview
1. **Image Loading**: Read images from local directories.
2. **Processing Pipeline**: Apply transformations like resizing, filtering, edge detection.
3. **Visualization**: Use Matplotlib to display original and processed images side-by-side.
4. **Extension Ready**: Easy to add new image operations or integrate with ML models.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/bradychin/ecoscan.git
    cd ecoscan
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script to execute the image processing pipeline:

```bash
python main.py
```

If no trained model exists at the configured path, training will begin.

If a trained model exists, the image predictor will launch immediately.