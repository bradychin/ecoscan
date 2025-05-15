# --------- Import Libraries ---------#
from keras import layers
import keras.applications.mobilenet_v2 as mobilenetv2

# --------- Mobile Net V2 Preprocessing Layer ---------#
class MobileNetV2PreprocessingLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(MobileNetV2PreprocessingLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return mobilenetv2.preprocess_input(inputs)

    def get_config(self):
        # This is required to ensure proper saving/loading
        return super().get_config()