import csv
from scipy.misc import imread
import numpy as np
import sklearn
from sample_generator import generator
import sys, getopt

inputFile = ''
outputFile = ''

def train(inputCSVFile, inputDir, outputDir):
    lines = []
    with open(inputCSVFile) as csvfile:
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

    train_generator = generator(train_samples, inputDir, batch_size=32)
    validation_generator = generator(validation_samples, inputDir, batch_size=32)

    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator,\
                steps_per_epoch = len(train_samples)/32,\
                validation_data = validation_generator,\
                validation_steps = len(validation_samples)/32,\
                epochs=5)

    # Save data
    import pickle
    outfilep = open(outputDir+'history.pkl', 'wb')
    pickle.dump(history, outfilep)
    outfilep.close()
    model.save(outputDir + 'model.h5')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hi:d:o:", ["help", "inputCSV=", "imageDir=", "outputDir="])
    except getopt.GetoptError:
        print ('train.py -i <inputCSV> -d <imageDir> -o <outputDir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print ('train.py -i <inputCSV> -d <imageDir> -o <outputDir>')
            sys.exit()
        if opt in ('-i', '--inputCSV'):
            print ('inputFile: ' + arg)
            inputFile = arg
        if opt in ('-d', '--imageDir'):
            print ('imageDir: ' + arg)
            inputDir = arg
        if opt in ('-o', '--outputDir'):
            print ('outputdir: ' + arg)
            outputDir = arg
    train(inputFile, inputDir, outputDir)

if __name__ == "__main__":
   main(sys.argv[1:])


