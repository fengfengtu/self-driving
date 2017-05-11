import os
import csv
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Cropping2D
from keras.layers import Convolution2D
import gc

# Read training data into samples

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

# Remove data with angles smaller than 1 degree

threshold = 1/90
samples = [x for x in samples if not (-threshold < float(x[3]) < threshold)]

#Define data augmentation methods

def random_change_brightness(image):
    # Change brightness randomly
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    image1[:,:,2] = image1[:,:,2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

def random_shift(image,steer,trans_range):
    # Shift the image horizontally and vertically randomly
    image_tr = np.array(image)
    tr_x = trans_range * (np.random.uniform() - 0.5)
    steer_ang = steer + tr_x/ trans_range * 2 * .25
    tr_y = 40 * (np.random.uniform() - 0.5)
    Trans_M = np.float32([[1,0,tr_x], [0,1,tr_y]])
    image_tr = cv2.warpAffine(image_tr, Trans_M,(image.shape[1], image.shape[0]))   
    return image_tr,steer_ang

def random_add_shadow(image):
    # Add random shadow to the image
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    if np.random.randint(2) == 1:
        random_bright = .8
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1] * random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0] * random_bright    
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return image

#Split the data into training data and validation data

train_samples, validation_samples = train_test_split(samples, test_size=0.1)

# define generator function to pull data on the fly and save memory
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:             
                angle = float(batch_sample[3])
                
                # create adjusted steering measurements for the side camera images
                correction = 0.07 # this is a parameter to tune
                
                #randomly choose left/center/right image
                which_image = np.random.randint(3)
                if which_image == 0:
                    name = './data/IMG/'+batch_sample[0].split('/')[-1]
                elif which_image == 1:
                    name = './data/IMG/'+batch_sample[1].split('/')[-1]
                    angle += correction
                else:
                    name = './data/IMG/'+batch_sample[2].split('/')[-1]
                    angle -= correction
                
                chosen_image = cv2.imread(name)
                
                # add shadow
                chosen_image = random_add_shadow(chosen_image)
                
                # resize to 200x100
                chosen_image = cv2.resize(chosen_image,(200,100), interpolation=cv2.INTER_AREA)
                
                # change brightness
                chosen_image = random_change_brightness(chosen_image)
                
                # convert to YUV color space
                chosen_image = cv2.cvtColor(chosen_image, cv2.COLOR_RGB2YCR_CB)        
                
                # horizontally/vertically shift the image
                chosen_image, angle =  random_shift(chosen_image, angle, 50)
                
                # randomly flip the image
                if np.random.randint(2) == 0:
                    chosen_image = np.fliplr(chosen_image)
                    angle = -angle
                
                images.append(chosen_image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# define model based on nvidia model 
model = Sequential()

model.add(Cropping2D(cropping=((33,1), (0,0)), input_shape=(100,200,3))) # -> (66,200,3)
model.add(Lambda(lambda x: x/255 - 0.5))
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))# -> (31,98,24)
#model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))# -> (14,47,36 )
#model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))# -> (5, 22, 48)
#model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))# -> (3, 18, 64)
#model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))# -> (1, 18, 64)
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch=10, verbose=1)

model.save('model.h5')
print ('Model trained and saved!')

gc.collect()
