import time
start = time.time()

import numpy as np
import pandas as pd
import pprint
import os
import random
import re
import nltk
import cv2
import matplotlib as mpl
mpl.use("TkAgg")

import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from random import shuffle
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from numpy.core.umath_tests import inner1d
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

percent_training = 0.85
n_estimators = 1000

for i in range(12, 67):
    ageVar = "age" + str(i) + "images = ''"
    ageList = "age" + str(i) + "test = ''"
    filesVar = "age" + str(i) + "files = [os.path.join(age" + str(i) + "images, file) for file in os.listdir(age" + str(i) + "images)]"
    forLoop = '''
    for j in range(int(len(age''' + str(i) + '''files)/10)):
        temp = random.choice(age''' + str(i) + '''files)
        age''' + str(i) + '''files.remove(age''' + str(i) + '''temp)
        age''' + str(i) + '''test.append(age''' + str(i) + '''temp)
    '''

    exec(ageVar)
    exec(filesVar)
    exec(forLoop)

# for j in range(int(len(not_age_files)/10)):
#     not_age_temp = random.choice(not_age_files)
#     not_age_files.remove(not_age_temp)
#     not_age_test.append(not_age_temp)

# initialize variable train_data with the 90% data

train_data = []
for i in age_files:
    train_data.append(i)
# for i in not_age_files:
#     train_data.append(i)

# initialize variable test_data with the 10% data

test_data = []
for i in age_test:
    test_data.append(i)
# for i in not_age_test:
#     test_data.append(i)

'''
print(train_data)
print(test_data)
'''

# function one_hot_label that creates variable ohl which determines if path is
# age_files or not_age_files

def one_hot_label(path):
    is_left = path.startswith(age_images) #may need revision, age_images might not exist
    ohl = np.array([1,0]) if is_left else np.array([0,1]) #else use label to find the 1 in the array
    return(ohl)

print(train_data)

# iterates through training data using OpenCV to read and resize the image, then
# adding it to list train_images with an array [0,1] or [1,0]

def train_data_with_label():
    train_images = []
    for path in tqdm(train_data):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64,64))
        train_images.append([np.array(img), one_hot_label(path)])
    shuffle(train_images) #randomize the list
    return(train_images)

# iterates through testing (new) data using OpenCV to read and resize the image,
# then adding it to list train_images with an array [0,1] or [1,0]

def test_data_with_label():
    test_images = []
    for path in tqdm(test_data):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64,64))
        test_images.append([np.array(img), one_hot_label(path)])
    return(test_images)

# initialize training images and testing images with above functions

training_images = train_data_with_label()
testing_images = test_data_with_label()

'''
test for functionality:
print(len(training_images))
print(len(testing_images))
'''

'''
# [len(img) for img in training_images]
'''

# initialize training data with labels and images, reshape function makes the array
# with dimensions 64, 64, 1, and first dimension is determined

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,64,64,1)
tr_lbl_data = np.array([i[1] for i in training_images])
tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,64,64,1)
tst_lbl_data = np.array([i[1] for i in testing_images])

#initialize Keras Sequential Model

model = Sequential()

model.add(InputLayer(input_shape = [64,64,1])) # tells model that input shape is 64,64,1, like above
model.add(Conv2D(filters=32, kernel_size=5,strides=1,padding='same',activation='relu')) # applies 2D convolution layer
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters=50,kernel_size=5,strides=5,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters=80,kernel_size=5,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(2,activation='softmax'))
optimizer=Adam(lr=1e-3)

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=tr_img_data,y=tr_lbl_data,epochs=50,batch_size=100)
model.summary()
print('Finished Model Training in ' + str(time.time() - start) + 's')


fig = plt.figure(figsize=(14,14))
for cnt, data in enumerate(testing_images): # enumerate puts each value into array
    y = fig.add_subplot(6,5,cnt+1) # add plots and coordinate to graph of image
    img = data[0]
    data = img.reshape(1,64,64,1) # reshape the dimensions of the image
    model_out = model.predict([data])

    if np.argmax(model_out) == 1: # if the max indice in array = 1, ___
        str_label='not_left'
    else:
        str_label='left'

    y.imshow(img,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show(block=True)
