import csv

# Read in the driving log csv file
# as shown in the "Training Yout Network" video
lines = []
with open('/opt/data/driving_log.csv') as csvfile:
    # Skip the line containing the table header
    next(csvfile)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Get the images and the steering measurements
import numpy as np
from scipy import ndimage

images = []
measurements = []
for line in lines:
    filename = '/opt/data/'+line[0]
    image = ndimage.imread(filename)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
# Convert the images and the steering measuremets to numpy arrays
# since this is the format Keras requires.
X_train = np.array(images)
y_train = np.array(measurements)

# Build the model using Keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
# Pre-processing using a lambda layer to improve the data:
# 1. Normalization: to a range between 0 and 1 by dividing by 255 - the max value of a pixel
# 2. Mean centering the data to shift the element mean by 0.5 down to zero
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

# Loss function is mean squared error (MSE) and the optimizer is ADAM
model.compile(loss='mse', optimizer='adam')
# The data is shuffled and 20% of it is saved for validation
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

# Save the trained model
model.save('model.h5')
