import csv

# Read in the driving log csv file
# as shown in the "Training Your Network" video
print('Reading the driving log csv... ', end='')
lines = []
with open('./data/driving_log.csv') as csvfile:
    # Skip the line containing the table header
    next(csvfile)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print('Done.')

# Get the images and the steering measurements
import numpy as np
from scipy import ndimage

print('Loading images... ', end='')
images = []
measurements = []
angle_correction = 0.2
for line in lines:
    for i in range(3):
        filename = './data/'+line[i].strip()
        image = ndimage.imread(filename)
        images.append(image)
        measurement = float(line[3])
        if i == 1:
            measurement += angle_correction
        elif i == 2:
            measurement -= angle_correction
        measurements.append(measurement)    
print('Done.')
print('Number of images in data set: {}'.format(len(images)))

# Data augmentation: flipped images to the data set
# if the steering angle's absolute value is greater than angle_threshold
angle_threshold = 0.02 # 1.0 means no augmentation
print('Augmenting data (steering angle threshold: {})... '.format(angle_threshold), end='')
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    if abs(measurement) > angle_threshold:
        augmented_images.append(np.fliplr(image))
        augmented_measurements.append(-measurement)
print('Done.')
print('Number of images in data set after augmentation: {}'.format(len(augmented_images)))
    
# Convert the images and the steering measuremets to numpy arrays
# since this is the format Keras requires.
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Build the model using Keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Dropout
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# Pre-processing the data using a lambda layer to improve the model in two simple steps:
# 1. Normalization to a range between 0 and 1 by dividing by 255 - the max value of a pixel
# 2. Mean centering the data by shifting the element mean by 0.5 down to zero
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

# Pre-processing again, cropping images
model.add(Cropping2D(cropping=((60, 30), (0, 0))))

# LeNet
model.add(Conv2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
#model.add(Dropout(0.5))
model.add(Dense(84))
#model.add(Dropout(0.5))
model.add(Dense(1))

# Loss function is mean squared error (MSE) and the optimizer is ADAM
model.compile(loss='mse', optimizer='adam')
# The data is shuffled and 20% of it is saved for validation
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10, batch_size=32)

# Save the trained model
model.save('model.h5')

print('All done.')
