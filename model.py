import os
import sys
#sys.path.append('D:\ProgramData\Anaconda3\envs\KKeras\Lib\site-packages')
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


#############################
print(os.path.abspath('__file__'))

lines = []
with open('./data/driving_log.csv') as csvfile:  # 相对路径，用./
    csvsheet = csv.reader(csvfile)
    for line in csvsheet:
        lines.append(line)

lines = lines[1:-1]  # Remove the title line
print(lines[0]) # show the first effective data, the structure should be 
                # [center pic], [left pic], [right pic], [steering],[throttle],[brake],[speed]
			
###############################
images = []
images_rz = []
measurements = []

train_samples, validation_samples = train_test_split(lines,test_size =0.2)

##############################

Cor_num = float(0.12); # 对左右相机的修正值

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images =[]
            angles =[]
            for batch_sample in batch_samples: # every row in set
                for i1 in range (0,2): # give back [0][1][2] column , Cannot use range(3)?!!!! why?!!!
                    name ='./data/IMG/'+batch_sample[i1].split('/')[-1] # the name of the pic [center][left][right]
                    #print(name + str(i1))
                    image_r = cv2.imread(name)
                    image_r2 = cv2.cvtColor(image_r,cv2.COLOR_BGR2RGB)
                    # Process left, right and flip
                    if i1 == 0:
                        center_angle = float(batch_sample[3])
                    else:
                        center_angle = float(batch_sample[3]) + (-2*Cor_num*i1 + 3*Cor_num)
                    for i2 in range(0,1):
                        if i2 ==1:
                            image_r2 = cv2.flip(image_r2,1)
                            center_angle = center_angle*-1
                    images.append(image_r2) # a set of image with [center][left][right]
                    angles.append(center_angle) #                                       
     
            X_train = np.array(images)
            Y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, Y_train)
            
train_generator = generator(train_samples,batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)
#############################
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D


#######Build Model######
model = Sequential()
#Pre-processing
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
# NIVIDIA End to End Learning for Self-Driving Cars， Figure（4）
# First Conv, 24*5*5
model.add(Conv2D(24, 5, strides=(2, 2), activation='relu'))
model.add(Dropout(0.7))
# Second Conv, 36*5*5
model.add(Conv2D(36, 5, strides=(2, 2), activation='relu'))
# Thrid Conv, 48*5*5
model.add(Conv2D(48, 5, strides=(2, 2), activation='relu'))
# Fourth Conv, 64*5*5
model.add(Conv2D(64, 3, activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Flatten())
# Four Fully-Connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

########################
#Run the Model
model.compile(loss = 'mse', optimizer = 'adam') #Regression问题，而不是Classification问

#history_object = model.fit(X_train, Y_train,validation_split = 0.2, shuffle = True, epochs = 5, verbose = 1)
history_object = model.fit_generator(train_generator, 
                                     steps_per_epoch = len(train_samples),
                                     validation_data = validation_generator,
                                     validation_steps = len(validation_samples), 
                                     epochs = 3,verbose = 1)
#history_object = model.fit_generator(train_generator, steps_per_epoch = len(train_samples),validation_data = validation_generator,nb_val_samples = len(validation_samples), nb_epoch = 5)


print(history_object.history.keys())
### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()

###########################

model.save('model.h5')
print('finish!')