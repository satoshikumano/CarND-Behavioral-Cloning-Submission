import numpy as np
from scipy.misc import imread
import sklearn

def generator(samples, inputDir, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            revises = [0.0, 0.2, -0.2]
            for batch_sample in batch_samples:
                for (i, rev) in zip (range(3), revises):
                    source_path = batch_sample[i]
                    source_path = source_path.strip()
                    filepath = inputDir + source_path
                    image = imread(filepath)
                    measurement = float(batch_sample[3])+rev
                    images.append(image)
                    measurements.append(measurement)
                    # Augment data : flipped image.
                    flip_image = np.fliplr(image)
                    flip_measurement = (measurement+rev) * -1.0
                    images.append(flip_image)
                    measurements.append(flip_measurement)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

