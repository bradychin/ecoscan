# --------- Import Libraries ---------#
import os

########################################################################
# Modify model path.
MODEL_PATH = 'models/model.keras'
########################################################################

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model_path = os.path.join(PROJECT_ROOT, MODEL_PATH)
dataset_path = os.path.join(PROJECT_ROOT, 'data/dataset')
model_weights = os.path.join(PROJECT_ROOT, 'models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')

batch_size = 64
image_size = 224, 224
class_mode = 'categorical'
color_mode = 'rgb'
epochs = 50
num_classes = 6