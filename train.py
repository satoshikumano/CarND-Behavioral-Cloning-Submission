import csv
from scipy.misc import imread
import numpy as np

inputDir = '/data/'
outputDir = '/output/'
#inputDir = 'data/'
#outputDir = 'output/'

lines = []
with open(inputDir + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines[1:]:
    source_path = line[0]
    filepath = inputDir + source_path
    image = imread(filepath)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images = []
augmented_measurements = []

from scipy import ndimage

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flip_image = np.fliplr(image)
    augmented_images.append(flip_image)
    augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save(outputDir+'model.h5')
