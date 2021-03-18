# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Figure_1.png "Center Cam Img Histogram"
[image2]: ./Figure_2.png "All Cam Img Histogram"
[image3]: ./Figure_3.png "Augmented Cam Img Histogram"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 video recording of my vehicle driving autonomously a little more than one lap around track 1 (using model.h5)
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around track 1 by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for loading and pre-processing the data, training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network. This is the LeNet architecture with 5x5 filter sizes and depths 6 (model.py lines 74-84) 

The model includes RELU layers to introduce nonlinearity (code lines 75 and 77), and the data is normalized in the model using a Keras lambda layer (code line 69). In the same lambda, the data is also mean centered. There is a further pre-processing step: cropping the images from size 160x320 to 70x320 - the upper 60 pixel rows and the bottom 30 pixel rows are removed from the images (code lines 71-72). So only the rows with important and relevant information in them remained in the images.

#### 2. Attempts to reduce overfitting in the model

First I tried to reduce overfitting by adding a dropout layer after both fully connected layer with drop out probabilities 0.5. But these were removed in the final model, as another strategy to reduce overfitting proved to be sufficient alone.

The other, successful strategy to reduce overfitting was to add more data: use the left and right-hand side camera images and augment the data set. For details about how I augmented the training data, see the section about Model Architecture and Training Strategy.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 89). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py lines 86-87).

#### 4. Appropriate training data

For training data the sample driving data was used as a basis.

For details about why and how I augmented the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My strategy for deriving a model architecture was to start with something very simple, and try to improve on it step-by-step by pre-processing the data, fine-tuning parameters and adding more neurons and layers to the network.

My first step was to use a network with no convolution layer in it, just a flatten and a dense layer similar to how it is shown in the Training Your Network video. I thought this would not be sufficient, but at least it was a good way to set up a simple pipeline, save the model, and try it out in the simulator. Obviously, it failed to provide the required result.

Then I built the already well known LeNet architecure.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that the LeNet model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To reduce overfitting, I added dropout layers and played around with drop out probabilities.

I also played around with batch sizes. With batch sizes higher than the default 32, the validation loss seemed to get consistently higher, so I left it at 32.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. The most interesting spot that I was usually waiting for in the simulator was the left curve after the bridge. Before the final model, my car/model always went off the track there and went straight on the dirt road instead or just bumped into an object on the side of the road and could not go further.

Then I started to use more (all) images from the data set and augmented the data.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

In the final model architecture the dropout layers were commented out. That was done first only in order to see what using all the images and data augmentation alone can bring to the table. But then, when I saw that that way alone the required results can be achieved, I did not put back the dropouts.

So the final model architecure is a LeNet-style architecture. It consists of the following layers and sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Lambda         		| input/output: 160x320x3                       |
| Cropping         		| outputs 70x320x3 - pre-processed image shape  |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 66x316x6   |
| RELU					| to introduce non-linearity					|
| Max pooling 2x2     	| 2x2 stride, valid padding, outputs 33x158x6	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 29x154x6   |
| RELU					| to introduce non-linearity 					|
| Max pooling 2x2     	| 2x2 stride, valid padding, outputs 14x77x6 	|
| Flatten       		| outputs 6468                 					|
| Fully connected		| outputs 120                 					|
| Fully connected		| outputs 84                 					|
| Fully connected		| outputs 1                 					|

#### 3. Creation of the Training Set & Training Process

I used the sample driving data as a starting point for creating the training and validation sets. In that data, images for driving straight (steering 0) are over-represented.

![alt text][image1]

In order to increase the volume of the training data to solve overfitting, I used all the images in the sample driving data: loaded the images recorded by the left- and righ-hand side cameras as well. This is a step I added to where I load the images (code lines 24-33). The steering angle measurements for the left- and right-hand side images were corrected by adding and subtracting the same correction value, respectively - yet another parameter to tune. In the final code this parameter was set to 0.2.

The representation of the steering values throughout the images became the following:

![alt text][image2]

In order to further increase the data and reduce over-fitting, I decided to do a simple augmentation of the data: simply flipped images with a steering angle greater than a threshold - also a parameter that was tuned (code lines 37-50). In the final code the angle threshold was set to 0.02.

![alt text][image3]

After loading all the images and augmenting the data, I had 43468 data points. I then pre-processed this data by normalization and mean centering pixel values and also by cropping the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. Around the 5th epoch the loss on the validation set seemed to vary a little around 0.02.

I used an adam optimizer so that manually training the learning rate wasn't necessary.
