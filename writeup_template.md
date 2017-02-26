#**Behavioral Cloning**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* data.py containing Sample class and related support functions to prepare samples to feed the generator
* nvidia.h5 containing a trained convolution neural network using the Nvidia model
* simple.h5 containing a trained convolution neural network based on a simpler model
* comma.h5 containing a trained convolution neural network based on the comma.ai model
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh python drive.py --speed 20 model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I tried 3 different models to tackle this project:
1. [NVIDIA](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) model, named as 'nvidia' within the code
2. comma.ai behavioral cloning model, named as 'comma' within the code
3. My own model based on a simplification of nvidia and comma models, named as 'simple' within the code

Each model has a different input size, so there is some image preprocessing required before the first layer to crop and resize the input image according to the selected model. Cropping removes 58 lines from the top and 22 lines from the bottom of the image. These are the shapes of the images during the preprocessing step:
IMAGE SHAPE: 160x320x3
CROPPED SHAPE: 80x320x3
INPUT SHAPES:
- 'nvidia' 66x200x3
- 'comma' 80x160x3
- 'simple' 40x40x3

All the models share the first Keras Lambda layer to normalize input values within the [-1, 1] range. I also tried max-min normalization with each image, but it did not make noticeable improvement due to the high contrast in the simulator images (like most video games). I then decided to keep the simple normalization (assuming min=0 and max=255) for performance reasons. If we had images recorded by a real camera, min-max normalization would make more sense to cope with low contrast images (e.g. night driving).

Models 'nvidia' and 'simple' include RELU activation layers to introduce nonlinearity. Model 'comma' includes ELU activation layers instead, as it is implemented in Comma's public [repository](https://github.com/commaai/research/blob/master/train_steering_model.py).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. The best value of dropout probability is quite difficult to find out, as it depends on the amount of data you have and the model size among others. At the end of the day the values I chose are initially based on intuition and fine-tuned based on the model performance. I used 0.4 and 0.2 for nvidia model and 0.2 for the simple model, only applied to the after layers with the most outputs.

I also tried L2 regularizer on the fully connected layers, but I saw not much benefit versus the dropout, so I decided to deactivate it for the sake of performance. Besides, when L2 regularization was activated, the optimizer convergence was much slower.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I added an early stopping callback to stop the training if the model is not improving its optimizer objective. I set the patience parameter to 4, defined as the number of epochs without improvement to allow before the training stops.

During the training phase, I only saved the model after the epoch with the minimum validation error, to ensure I end up with the model with less overfitting.

####3. Model parameter tuning

The model used an adam optimizer, which automatically tunes the learning rate at each epoch. I tried with several learning rate initialization values, but finally left Adam default value (0.001).

I also played with different batch sizes, ending up with 256 as a good compromise between quick and precise convergence of the loss function.

I set the maximum number of epochs to 20, mainly due to the fact I have limited GPU resources (AWS) and cannot run very long training sessions. Ideally, I would have liked to try with 100 epochs along with much more data.

####4. Appropriate training data

This ended up as the most important factor for a precise learning, even more than the model selection. Even though I was able to successfully train the model just with the data provided by Udacity, I decided to make my own dataset for the sake of learning. I used a combination of center lane driving, recovering from the left and right sides of the road.

Collecting the recovery data correctly, was the most challenging part: At first, the V1 simulator required to press the mouse button to record samples, so moving one hand out of the joystick to click the mouse on the record button while keeping the car on the track was not easy at all. Then the V2 simulator was released, but it did not support the joystick due to a bug. Using the keyboard was clearly a limiting factor to provide with precise steering angles. Finally the bug was solved, but it still was difficult to press the R key at the right time while using the joystick. If I find the right time, I will consider a pull request to activate/deactivate recording with a joystick button, which I believe can notably ease this whole data collection process.

For details about how I created the training data, see the next section.

###Architecture and Training Documentation

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
