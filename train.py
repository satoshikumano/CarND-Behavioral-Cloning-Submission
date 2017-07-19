import csv
from scipy.misc import imread
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                filepath = inputDir + source_path
                image = imread(filepath)
                measurement = float(batch_sample[3])
                images.append(image)
                measurements.append(measurement)
                # Augment data : flipped image.
                flip_image = np.fliplr(image)
                flip_measurement = measurement * -1.0
                images.append(flip_image)
                measurements.append(flip_measurement)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

inputDir = '/data/'
outputDir = '/output/'
#inputDir = 'data/'
#outputDir = 'output/'

lines = []
with open(inputDir + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from scipy import ndimage

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

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,\
            steps_per_epoch = len(train_samples)/32,\
            validation_data = validation_generator,\
            validation_steps = len(validation_samples)/32,\
            epochs=5)

model.save(outputDir+'model.h5')
