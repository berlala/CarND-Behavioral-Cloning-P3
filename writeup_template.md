# **Behavioral Cloning** 

Bolin Zhao

E-mail: bolinzhao@yahoo.com

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./pics/nvidia_net.png "Model Visualization"
[image2]: ./pics/fork.png "Fork road"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"



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

The model is reference to the well-known Nvidia research work  in <End to End Learning for Self-Driving Cars>. The  structure is shown as below, it consist of 5 convolutional layers, 1 flatten layer and 3 fully-connected layers. Since in this project, the output is only a number, the final neurons is only 1 which represent the steering angle.

![the Nvidia CNN model][image1]



#### 2. Attempts to reduce overfitting in the model

The model contains one dropout layers with rate 0.7 in order to reduce overfitting. One dropout layer is proofed to work on track1.

#### 3. Model parameter tuning

The model used an adam optimizer.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. In this project, the Udaicty provided sample data is used as the resource. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

My first step was to use a convolution neural network model similar to the  NVIDIA paper since this paper present a solution to a similar problem.

The first training is taken by the whole train set and validate by the validation set which is 20% of the whole set.  The epoch is set to 5. and the result is quite good.  The car seems can stay on the track for most of time.

To follow the instruction on the course, a generator is design to process the whole training data. The generator can reduce the usage of the memory during training meanwhile add the random into the training process. The validation set is still 20% of the whole set. The generator then load the training set by defined batch size.  The epoch is set to 2. But the result is not good, although the final validation loss is  0.0157, the car cannot pass the first left turn and get out of the track.  

Then the left an right picture from the left and right camera are used to enhance the performance.  A correlation value is set as +0.1 for left picture(since you want the car to turn a little bit to right if in this position) and -0.1 for right picture(model.py line 57~60). The epoch is 3 and the final validation loss is 0.0112. But the car is still cannot finish the whole track. It turn not big enough in the  fork way(show in the following picture).

![The fork road][image2]



The final try is to add the flip pictures into the data set (model.py line 57~60). The training epoch is still 3. And finally the car can drive on the track quite well. But in the fork way above, it is still obvious turning not big enough. If it is possible, more pics around this corner should be added into the training set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road on track1.


