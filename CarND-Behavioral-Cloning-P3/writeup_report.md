# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_lane_driving.png "center lane driving"
[image3]: ./examples/recover1.jpg "Recovery Image"
[image4]: ./examples/recover2.jpg "Recovery Image"
[image5]: ./examples/recover3.jpg "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./examples/orig_distr.png "Image"
[image9]: ./examples/after_distr.png "Image"
[image10]: ./examples/center_image.png "Image"
[image11]: ./examples/center_image_flipped.png "Image"
[image12]: ./examples/center_image1.png "Image"
[image13]: ./examples/center_image1_bright.png "Image"
[image14]: ./examples/center_image2.png "Image"
[image15]: ./examples/center_image2_shadow.png "Image"
[image16]: ./examples/center_image3.png "Image"
[image17]: ./examples/center_image3_trans.png "Image"
[image18]: ./examples/loss_graph.png "Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the nvidia model (https://arxiv.org/abs/1604.07316), which consists of a convolution neural network with 5 convolutional layers and 3 fully connected layers. (model.py lines 137-154) 

The first three convolutional layers use a 5x5 filter and a 2x2 stride, and the remaining two convolutional layers use a 3x3 filter and 1x1 stride.

The model includes RELU layers to introduce nonlinearity (code lines 139-154), and the data is normalized in the model using a Keras lambda layer (code line 138). The model also includes a cropping layer to remove the top unwanted section of the image.

#### 2. Attempts to reduce overfitting in the model

I tested the model with and without the dropout layers (model.py lines 140, 142, 144, 146, 148). Because the data augmentation is also a way to avoid overfitting, in this project, I didn't find overfitting problem with or without the drop layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 71). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 156).

#### 4. Appropriate training data

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a tried and proven network such as LeNet 5, AlexNet, VGG, etc, and build upon it. I then come across the nvidia network which was used in a very similar scenario and the network is a simple one, so I decided to give it a try.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

I ran the simulator to see how well the car was driving around track one. It drove ok until it fell off the track at the relatively sharp turn after the bridge. To improve the driving behavior in these cases, I chose a bunch of data augmentation methods such as adding the left/right camera images and tuning the angle correction parameter, adding the horizontal/vertical shift of the image, adding random shadow and changing the brightness of the images. After the data augmentation, the model runs quite well.

I also experimented with adding a dropout layer with 0.50 prob after each convolutional layer but there's no noticeable improvement both in terms of loss and the simulation.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 137-154) consisted of a convolution neural network with the following layers and layer sizes:



| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 100x200x3 YUV image   						| 
| Cropping				| cropping the top part to 66x200x3 YUV image	|
| Lamda					| Normalize the pixel value to (-0.5, 0.5)			|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 31x98x24 	|
| RELU					| non-linear transformation				        |
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 14x47x36 	|
| RELU					| non-linear transformation						|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 5x22x48 	|
| RELU					| non-linear transformation						|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 3x20x64 	|
| RELU					| non-linear transformation						|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 1x18x64 	|
| RELU					| non-linear transformation						|
| Flatten				| 												|
| Fully connected		| 1164        									|
| RELU					| non-linear transformation						|
| Fully connected		| 100        									|
| RELU					| non-linear transformation						|
| Fully connected		| 50        									|
| RELU					| non-linear transformation						|
| Fully connected		| 10        									|
| RELU					| non-linear transformation						|
| Fully connected		| 1        								    	|
| TANH					| non-linear transformation					    |
 

#### 3. Creation of the Training Set & Training Process

I examined the udacity training data and decided to use it with data augmentation. Here is an example image of center lane driving:

![alt text][image2]

I also examined the data where the vehicle recovered from the right sides of the road back to center so that the vehicle would learn to drive along the center. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

I drew a histogram of the steering angles in the training data and found a disproportional part of the data is small angles centered around 0. It would cause the model to bias toward 0, therefore, I removed the data with angles smaller than 1. The original distribution and after distribution are as follows.

![alt text][image8]
![alt text][image9]

To augment the data sat, I also flipped images and angles thinking that this would balance the training data such that it does not skew toward either left or right turns. For example, here is an image that has then been flipped:

![alt text][image10]
![alt text][image11]

I also did the following augmenations to make the model more robust. 

#### 1. Randomly change the brightness of the image.

![alt text][image12]
![alt text][image13]

#### 2. Randomly add shadow to the image:

![alt text][image14]
![alt text][image15]

#### 3. Randomly horizontally/vertically shift the image .

![alt text][image16]
![alt text][image17]


After the collection process, I had 3584 number of data points. I then preprocessed this data by resizing the image to the shape (100,200,3) and convert it to YUV color space.

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by the following graph. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image18]
