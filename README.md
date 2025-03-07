# EcoScan
EcoScan is a computer vision-based project designed to classify recyclable materials using deep learning. The project 
leverages the TrashNet dataset to train a model that can accurately categorize waste into different recyclable categories.

## Programs
1. ecoscan.py: Build, train, and evaluate the deep learning model.
2. evaluate.py: Make predictions from a single image by loading in model and user defined image.

### Instructions to run ecoscan.py:
1. Ensure that folder "dataset-resized" is unzipped and in the same folder as ecoscan.py.
2. Ensure that file "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5" is in the same folder as ecoscan.py
3. Run ecoscan.py to create model. 

Note: The model has already been created (model.keras.). Only run ecoscan.py if needed.

### Instructions to run evaluate.py:
1. Ensure that "ecoscan.py" and ""model.keras" are in the same folder as evaluate.py.
2. Ensure that the image you want to predict is in .jpg format.
3. Place you image into the same folder as evaluate.py.
4. Run evaluate.py to make prediction.
5. Enter the image name, including the file extension, when prompted.

Note: Test images are provided. The above instructions still apply

