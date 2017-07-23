import csv
from scipy.misc import imread
import numpy as np
import sklearn
from sample_generator import generator
import sys, getopt

def train(inputCSVFile, inputDir, inputModel, outputDir):
    lines = []
    with open(inputCSVFile) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    from scipy import ndimage

    from keras.models import Sequential, load_model
    from keras.layers import Flatten, Dense, Lambda, Dropout
    from keras.layers.convolutional import Conv2D, Cropping2D
    from keras.layers.pooling import MaxPooling2D
    from keras.backend import tf as ktf

    if inputModel == "":
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
        model.add(Cropping2D(cropping=((70,25),(0,0))))
        model.add(Conv2D(24, (5, 5), strides=(2,2), activation="relu"))
        model.add(Conv2D(36, (5, 5), strides=(2,2), activation="relu"))
        model.add(Conv2D(48, (5, 5), strides=(2,2), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
    else:
        print ('load model: ' + inputModel)
        model = load_model(inputModel)

    from sklearn.model_selection import train_test_split
    #train_samples, validation_samples = train_test_split(lines[1:], train_size=0.4, test_size=0.1)
    train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)

    train_generator = generator(train_samples, inputDir, batch_size=32)
    validation_generator = generator(validation_samples, inputDir, batch_size=32)

    history = model.fit_generator(train_generator,\
                steps_per_epoch = len(train_samples)/32,\
                validation_data = validation_generator,\
                validation_steps = len(validation_samples)/32,\
                epochs=5)

    # Save model
    model.save(outputDir + 'model.h5')
    # Save hisotry as graph
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(outputDir+'history.png')

def main(argv):
    helpmsg = 'train.py -i <inputCSV> -m <inputModel> -d <imageDir> -o <outputDir>'
    try:
        opts, args = getopt.getopt(argv, "hi:m:d:o:", ["help", "inputCSV=", "inputModel=", "imageDir=", "outputDir="])
    except getopt.GetoptError:
        print (helpmsg)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print (helpmsg)
            sys.exit()
        if opt in ('-i', '--inputCSV'):
            print ('inputFile: ' + arg)
            inputFile = arg
        if opt in ('-m', '--inputModel'):
            print ('inputModel: ' + arg)
            if arg != 'none':
                inputModel = arg
            else:
                inputModel = ""
        if opt in ('-d', '--imageDir'):
            print ('imageDir: ' + arg)
            inputDir = arg
        if opt in ('-o', '--outputDir'):
            print ('outputdir: ' + arg)
            outputDir = arg
    train(inputFile, inputDir, inputModel, outputDir)

if __name__ == "__main__":
   main(sys.argv[1:])


