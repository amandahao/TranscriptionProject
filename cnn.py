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

ageImages = "./ages"

testList = []
filesList = []

for i in range(12, 67):
    filesList = [os.path.join(ageImages, file) for file in os.listdir(ageImages)]

    # filesList[str(i)] = [os.path.join(ageimagesdict[str(i)], file) for file in os.listdir(ageimagesdict[str(i)])

for i in range(int(len(filesList)/10)):
    temp = random.choice(filesList)
    filesList.remove(temp)
    testList.append(temp)

# initialize variable trainData with the 90% data

trainData = []
for i in filesList:
    trainData.append(i)
# for i in not_age_files:
#     trainData.append(i)

# initialize variable testData with the 10% data

testData = []
for i in testList:
    testData.append(i)
# for i in not_age_test:
#     testData.append(i)

'''
print(trainData)
print(testData)
'''

# function one_hot_label that creates variable ohl which determines if path is
# age_files or not_age_files

def genArray(age, label):
    # right? = path.startswith("age18")
    print(age, end=" ")
    print(label)

    rightArray = []
    wrongArray = []
    if age == label:
        for i in range(12, 67):
          if age == i:
            rightArray.append(1)
          else:
            rightArray.append(0)
        return rightArray
    else:
        for i in range(12, 67):
            x = 0
            if label == i:
                wrongArray.append(1)
            else:
                wrongArray.append(0)
        return wrongArray

# def one_hot_label(path):
#     is_left = path.startswith("age18") #may need revision, age_images might not exist
#     ohl = np.array([1,0]) if is_left else np.array([0,1]) #else use label to find the 1 in the array
#     return(ohl)

print(trainData)

# iterates through training data using OpenCV to read and resize the image, then
# adding it to list train_images with an array [0,1] or [1,0]

def trainDataWithLabel():
    train_images = []
    for path in tqdm(trainData):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64,64))
        for i in range(12, 67):
            train_images.append([np.array(img), genArray(i, str(path)[10:12])])
        # train_images.append([np.array(img), one_hot_label(path)]) #genArray(age, str(path)[3:5])
    shuffle(train_images) #randomize the list
    return(train_images)

# iterates through testing (new) data using OpenCV to read and resize the image,
# then adding it to list train_images with an array [0,1] or [1,0]

def testDataWithLabel():
    test_images = []
    for path in tqdm(testData):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64,64))
        for i in range(12, 67):
            test_images.append([np.array(img), genArray(i, str(path)[10:12])])
    return(test_images)

# initialize training images and testing images with above functions

training_images = trainDataWithLabel()
testing_images = testDataWithLabel()
print("1")
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
print("2")

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
model.add(Dense(55,activation='softmax'))
optimizer=Adam(lr=1e-3)
print("3")

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=tr_img_data,y=tr_lbl_data,epochs=50,batch_size=100)
print("4")

model.summary()

print("5")

print('Finished Model Training in ' + str(time.time() - start) + 's')
# print(model.summary())
print("6")


fig = plt.figure(figsize=(14,14))
for cnt, data in enumerate(testing_images): # enumerate puts each value into array
    y = fig.add_subplot(1000,1000,cnt+1) # add plots and coordinate to graph of image
    img = data[0]
    data = img.reshape(1,64,64,1) # reshape the dimensions of the image
    model_out = model.predict([data])
    print("7")

    print(model_out)
    # if np.argmax(model_out) == 1: # if the max indice in array = 1, ___
    #     str_label='not_left'
    # else:
    #     str_label='left'

    # y.imshow(img,cmap='gray')
    # plt.title(str_label)
    # y.axes.get_xaxis().set_visible(False)
    # y.axes.get_yaxis().set_visible(False)

# plt.show(block=True)

testImg = cv2.imread(testImages[1], cv2.IMREAD_GRAYSCALE)
testImg = cv2.resize(testImg, (64,64))
testImg = np.array(testImg)
model.predict(testImg)
