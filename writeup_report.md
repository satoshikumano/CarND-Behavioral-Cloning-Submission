# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[dirt road]: ./resources/dirt-road.jpg "Dirt road"

### Files Submitted

My project includes the following files:

* [model.py](./model.py) containing the script to create and train the model.
* [sample\_generator.py](./sample_generator.py) separated generator function for the readability.
* [drive.py](./drive.py) for driving the car in autonomous mode.
* [model.h5](./model.h5) containing a trained convolution neural network.
* [writeup\_report.md](./writeup_report.md) summarizing the results.

### Model Architecture and Training Strategy (With Le-Net)

I started with Le-Net architecture.
Got better result with different architecture mentioned later.
However, it was possible to train the network and able to go around the course1.

#### 1. Image processing

- Normalization: Normalize RGB values with x / 255.0 - 0.5.
- Cropping: vertically crop 70 pixel from the top, 25 pixel from the bottom.

#### 2. Augmentation

- Generates mirror image using numpy.fliplr().
  Steering angle for flipped image is original angle * -1.

#### 3. Architecture

Architecture is described as following.

```python
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
```

#### 4. Training data

I started by the [sample data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)
provided by Udacity.
I only used center image for the Le-Net version.

#### 5. Tuning

No learning late tuning since I chose Adam optimizer.

Observed overfitting on the first training.
Added Dropout to fully connected layers to mitigate overfitting.

[code](https://github.com/satoshikumano/CarND-Behavioral-Cloning-Submission/blob/le-net/model.py#L32-L34)

#### 6. Result

After the training, run the simulator in autonomous mode.
Can't finish the lap. The car can't back to the center when it approaches to
the edge of the road.

#### 7. More training

Run the simulator in manual mode and recorded sharp turn on the edge.
Load the saved 'model.h5' generated on step 6 
and train the network with the additional recordings.

Result: Getting better. Now car can go back to the center when it approaches
to the edge of the raod.
However, it chooses ![alt text][dirt road].

#### 8. More training 2

Run the simulator in manual mode and recorded driving near 
![alt text][dirt road] point repeatedly.

Load the saved 'model.h5' generated on step 7.
and train the network with the additional recordings.

Result: Works well. Now the car can around the lap.
However, the car is weaving a lot after the step 7.

[Driving video](./videos/lenet-fin.mp4)

### Model Architecture and Training Strategy (With Nvidia pipeline)

With Le-Net, It is possible to to drive around the course 1.
However, It is too sensitive.
For example, I tried to smoother the steering after the step 8 in previous section.
I recorded additional laps steering smoother and train the network inheriting the
saved model on step 8.
After the additional training, Steering gets smoother but can't go back to the
center when the car approaches to the edge.
The result was similar to step 6 even after the additional learning.

I decided to use more powerful architecture descrived in the
[paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
published by Nvidia to see the difference.

#### 1. Image processing

- Normalization: Normalize RGB values with x / 255.0 - 0.5.
- Cropping: vertically crop 70 pixel from the top, 25 pixel from the bottom.

#### 2. Augmentation

- Generates mirror image using numpy.fliplr().
  Steering angle for flipped image is original angle * -1.


#### 3. Architecture

Architecture is described as following.

```python
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
```

#### 4. Training data

I started by the [sample data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)
provided by Udacity.
Used only center image for the first trial.

#### 5. Tuning

No learning late tuning since I chose Adam optimizer.
Tuning was not necessary since overfitting/ underfitting was not observed.

#### 6. Result

After the training, run the simulator in autonomous mode.
The car go around the course 1.

[Driving video](./videos/nvidia-center.mp4)

#### 7. Use left/ right camera images.

In the step 6, the car can go around the course 1.
Let's see using left/ right image can improve the driving.
Steering angle for left image is original angle + 0.2, right image is orignal angle - 0.2.

Result:

After the training, run the simulator in autonomous mode.
The car go around the course 1.
Bit weaving comparing to the result not using left/ right images.

[Driving video](./videos/nvidia-lr.mp4)

### Final model submitted:

[model.h5](./model.h5) is generated by following system/ condition.

#### Architecture:
- Nvidia pipeline

#### Training data:
- Sample provided by udacity. (using only center camera images.)

#### Image processing:

- Normalization: Normalize RGB values with x / 255.0 - 0.5.
- Cropping: vertically crop 70 pixel from the top, 25 pixel from the bottom.

#### Data augmentation

- Generates mirror image using numpy.fliplr().
  Steering angle for flipped image is original angle * -1.