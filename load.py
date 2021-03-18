# Build the model using Keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Dropout
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.models import load_model

model = load_model('model.h5')

model.summary()
