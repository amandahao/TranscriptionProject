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

age12images = ''
age13images = ''
age14images = ''
age15images = ''
age16images = ''
age17images = ''
age18images = ''
age19images = ''
age20images = ''
age21images = ''
age22images = ''
age23images = ''
age24images = ''
age25images = ''
age26images = ''
age27images = ''
age28images = ''
age29images = ''
age30images = ''
age31images = ''
age32images = ''
age33images = ''
age34images = ''
age35images = ''
age36images = ''
age37images = ''
age38images = ''
age39images = ''
age40images = ''
age41images = ''
age42images = ''
age43images = ''
age44images = ''
age45images = ''
age46images = ''
age47images = ''
age48images = ''
age49images = ''
age50images = ''
age51images = ''
age52images = ''
age53images = ''
age54images = ''
age55images = ''
age56images = ''
age57images = ''
age58images = ''
age59images = ''
age60images = ''
age61images = ''
age62images = ''
age63images = ''
age64images = ''
age65images = ''
age66images = ''
# right_images = './imageSet/right'
# up_images = './imageSet/up'

# initialize the image data

age12files = [os.path.join(age12images, file) for file in os.listdir(age12images)]
age13files = [os.path.join(age13images, file) for file in os.listdir(age13images)]
age14files = [os.path.join(age14images, file) for file in os.listdir(age14images)]
age15files = [os.path.join(age15images, file) for file in os.listdir(age15images)]
age16files = [os.path.join(age16images, file) for file in os.listdir(age16images)]
age17files = [os.path.join(age17images, file) for file in os.listdir(age17images)]
age18files = [os.path.join(age18images, file) for file in os.listdir(age18images)]
age19files = [os.path.join(age19images, file) for file in os.listdir(age19images)]
age20files = [os.path.join(age20images, file) for file in os.listdir(age20images)]
age21files = [os.path.join(age21images, file) for file in os.listdir(age21images)]
age22files = [os.path.join(age22images, file) for file in os.listdir(age22images)]
age23files = [os.path.join(age23images, file) for file in os.listdir(age23images)]
age24files = [os.path.join(age24images, file) for file in os.listdir(age24images)]
age25files = [os.path.join(age25images, file) for file in os.listdir(age25images)]
age26files = [os.path.join(age26images, file) for file in os.listdir(age26images)]
age27files = [os.path.join(age27images, file) for file in os.listdir(age27images)]
age28files = [os.path.join(age28images, file) for file in os.listdir(age28images)]
age29files = [os.path.join(age29images, file) for file in os.listdir(age29images)]
age30files = [os.path.join(age30images, file) for file in os.listdir(age30images)]
age31files = [os.path.join(age31images, file) for file in os.listdir(age31images)]
age32files = [os.path.join(age32images, file) for file in os.listdir(age32images)]
age33files = [os.path.join(age33images, file) for file in os.listdir(age33images)]
age34files = [os.path.join(age34images, file) for file in os.listdir(age34images)]
age35files = [os.path.join(age35images, file) for file in os.listdir(age35images)]
age36files = [os.path.join(age36images, file) for file in os.listdir(age36images)]
age37files = [os.path.join(age37images, file) for file in os.listdir(age37images)]
age38files = [os.path.join(age38images, file) for file in os.listdir(age38images)]
age39files = [os.path.join(age39images, file) for file in os.listdir(age39images)]
age40files = [os.path.join(age40images, file) for file in os.listdir(age40images)]
age41files = [os.path.join(age41images, file) for file in os.listdir(age41images)]
age42files = [os.path.join(age42images, file) for file in os.listdir(age42images)]
age43files = [os.path.join(age43images, file) for file in os.listdir(age43images)]
age44files = [os.path.join(age44images, file) for file in os.listdir(age44images)]
age45files = [os.path.join(age45images, file) for file in os.listdir(age45images)]
age46files = [os.path.join(age46images, file) for file in os.listdir(age46images)]
age47files = [os.path.join(age47images, file) for file in os.listdir(age47images)]
age48files = [os.path.join(age48images, file) for file in os.listdir(age48images)]
age49files = [os.path.join(age49images, file) for file in os.listdir(age49images)]
age50files = [os.path.join(age50images, file) for file in os.listdir(age50images)]
age51files = [os.path.join(age51images, file) for file in os.listdir(age51images)]
age52files = [os.path.join(age52images, file) for file in os.listdir(age52images)]
age53files = [os.path.join(age53images, file) for file in os.listdir(age53images)]
age54files = [os.path.join(age54images, file) for file in os.listdir(age54images)]
age55files = [os.path.join(age55images, file) for file in os.listdir(age55images)]
age56files = [os.path.join(age56images, file) for file in os.listdir(age56images)]
age57files = [os.path.join(age57images, file) for file in os.listdir(age57images)]
age58files = [os.path.join(age58images, file) for file in os.listdir(age58images)]
age59files = [os.path.join(age59images, file) for file in os.listdir(age59images)]
age60files = [os.path.join(age60images, file) for file in os.listdir(age60images)]
age61files = [os.path.join(age61images, file) for file in os.listdir(age61images)]
age62files = [os.path.join(age62images, file) for file in os.listdir(age62images)]
age63files = [os.path.join(age63images, file) for file in os.listdir(age63images)]
age64files = [os.path.join(age64images, file) for file in os.listdir(age64images)]
age65files = [os.path.join(age65images, file) for file in os.listdir(age65images)]
age66files = [os.path.join(age66images, file) for file in os.listdir(age66images)]
# right_files = [os.path.join(right_images, file) for file in os.listdir(right_images)]
# up_files = [os.path.join(up_images, file) for file in os.listdir(up_images)]
# not_age_files = right_files + up_files

'''
print(len(age_files))
print(len(right_files))
print(len(up_files))
print(len(not_age_files))
'''

# initialize  the lists for the training data

age_test = []
# not_age_test = []

# separate 10% of the image data from age_files and not_age_files for training


for j in range(int(len(age12files)/10)):
    age12temp = random.choice(age12files)
    age12files.remove(age12temp)
    age12test.append(age12temp)

for j in range(int(len(age13files)/10)):
    age13temp = random.choice(age13files)
    age13files.remove(age13temp)
    age13test.append(age13temp)

for j in range(int(len(age14files)/10)):
    age14temp = random.choice(age14files)
    age14files.remove(age14temp)
    age14test.append(age14temp)

for j in range(int(len(age15files)/10)):
    age15temp = random.choice(age15files)
    age15files.remove(age15temp)
    age15test.append(age15temp)


for j in range(int(len(age16files)/10)):
    age16temp = random.choice(age16files)
    age16files.remove(age16temp)
    age16test.append(age16temp)

for j in range(int(len(age17files)/10)):
    age17temp = random.choice(age17files)
    age17files.remove(age17temp)
    age17test.append(age17temp)


for j in range(int(len(age18files)/10)):
    age18temp = random.choice(age18files)
    age18files.remove(age18temp)
    age18test.append(age18temp)

for j in range(int(len(age19files)/10)):
    age19temp = random.choice(age19files)
    age19files.remove(age19temp)
    age19test.append(age19temp)


for j in range(int(len(age20files)/10)):
    age20temp = random.choice(age20files)
    age20files.remove(age20temp)
    age20test.append(age20temp)

for j in range(int(len(age21files)/10)):
    age21temp = random.choice(age21files)
    age21files.remove(age21temp)
    age21test.append(age21temp)


for j in range(int(len(age22files)/10)):
    age22temp = random.choice(age22files)
    age22files.remove(age22temp)
    age22test.append(age22temp)

for j in range(int(len(age12files)/10)):
    age23temp = random.choice(age23files)
    age23files.remove(age23temp)
    age23test.append(age23temp)


for j in range(int(len(age24files)/10)):
    age24temp = random.choice(age24files)
    age24files.remove(age24temp)
    age24test.append(age24temp)

for j in range(int(len(age25files)/10)):
    age25temp = random.choice(age25files)
    age25files.remove(age25temp)
    age25test.append(age25temp)


for j in range(int(len(age26files)/10)):
    age26temp = random.choice(age26files)
    age26files.remove(age26temp)
    age26test.append(age26temp)

for j in range(int(len(age27files)/10)):
    age27temp = random.choice(age27files)
    age27files.remove(age27temp)
    age27test.append(age27temp)


for j in range(int(len(age28files)/10)):
    age28temp = random.choice(age28files)
    age28files.remove(age28temp)
    age28test.append(age28temp)

for j in range(int(len(age29files)/10)):
    age29temp = random.choice(age29files)
    age29files.remove(age29temp)
    age29test.append(age29temp)


for j in range(int(len(age30files)/10)):
    age30temp = random.choice(age30files)
    age30files.remove(age30temp)
    age30test.append(age30temp)

for j in range(int(len(age31files)/10)):
    age31temp = random.choice(age31files)
    age31files.remove(age31temp)
    age31test.append(age31temp)


for j in range(int(len(age32files)/10)):
    age32temp = random.choice(age32files)
    age32files.remove(age32temp)
    age32test.append(age32temp)

for j in range(int(len(age33files)/10)):
    age33temp = random.choice(age33files)
    age33files.remove(age33temp)
    age33test.append(age33temp)


for j in range(int(len(age34files)/10)):
    age34temp = random.choice(age34files)
    age34files.remove(age34temp)
    age34test.append(age34temp)

for j in range(int(len(age35files)/10)):
    age35temp = random.choice(age35files)
    age35files.remove(age35temp)
    age35test.append(age35temp)

for j in range(int(len(age36files)/10)):
    age36temp = random.choice(age36files)
    age36files.remove(age36temp)
    age36test.append(age36temp)


for j in range(int(len(age37files)/10)):
    age37temp = random.choice(age37files)
    age37files.remove(age37temp)
    age37test.append(age37temp)

for j in range(int(len(age38files)/10)):
    age38temp = random.choice(age38files)
    age38files.remove(age38temp)
    age38test.append(age38temp)


for j in range(int(len(age39files)/10)):
    age39temp = random.choice(age39files)
    age39files.remove(age39temp)
    age39test.append(age39temp)

for j in range(int(len(age40files)/10)):
    age40temp = random.choice(age40files)
    age40files.remove(age40temp)
    age40test.append(age40temp)


for j in range(int(len(age41files)/10)):
    age41temp = random.choice(age41files)
    age41files.remove(age41temp)
    age41test.append(age41temp)

for j in range(int(len(age42files)/10)):
    age42temp = random.choice(age42files)
    age42files.remove(age42temp)
    age42test.append(age42temp)

for j in range(int(len(age43files)/10)):
    age43temp = random.choice(age43files)
    age43files.remove(age43temp)
    age43test.append(age43temp)

for j in range(int(len(age44files)/10)):
    age44temp = random.choice(age44files)
    age44files.remove(age44temp)
    age44test.append(age44temp)


for j in range(int(len(age45files)/10)):
    age45temp = random.choice(age45files)
    age45files.remove(age45temp)
    age45test.append(age45temp)

for j in range(int(len(age46files)/10)):
    age46temp = random.choice(age46files)
    age46files.remove(age46temp)
    age46test.append(age46temp)


for j in range(int(len(age47files)/10)):
    age47temp = random.choice(age47files)
    age47files.remove(age47temp)
    age47test.append(age47temp)

for j in range(int(len(age48files)/10)):
    age48temp = random.choice(age48files)
    age48files.remove(age48temp)
    age48test.append(age48temp)


for j in range(int(len(age49files)/10)):
    age49temp = random.choice(age49files)
    age49files.remove(age49temp)
    age49test.append(age49temp)

for j in range(int(len(age50files)/10)):
    age50temp = random.choice(age50files)
    age50files.remove(age50temp)
    age50test.append(age50temp)


for j in range(int(len(age51files)/10)):
    age51temp = random.choice(age51files)
    age51files.remove(age51temp)
    age51test.append(age51temp)

for j in range(int(len(age52files)/10)):
    age52temp = random.choice(age52files)
    age52files.remove(age52temp)
    age52test.append(age52temp)


for j in range(int(len(age53files)/10)):
    age53temp = random.choice(age53files)
    age53files.remove(age53temp)
    age53test.append(age53temp)

for j in range(int(len(age54files)/10)):
    age54temp = random.choice(age54files)
    age54files.remove(age54temp)
    age54test.append(age54temp)


for j in range(int(len(age55files)/10)):
    age55temp = random.choice(age55files)
    age55files.remove(age55temp)
    age55test.append(age55temp)

for j in range(int(len(age56files)/10)):
    age56temp = random.choice(age56files)
    age56files.remove(age56temp)
    age56test.append(age56temp)


for j in range(int(len(age57files)/10)):
    age57temp = random.choice(age57files)
    age57files.remove(age57temp)
    age57test.append(age57temp)

for j in range(int(len(age58files)/10)):
    age58temp = random.choice(age58files)
    age58files.remove(age58temp)
    age58test.append(age58temp)


for j in range(int(len(age59files)/10)):
    age59temp = random.choice(age59files)
    age59files.remove(age59temp)
    age59test.append(age59temp)

for j in range(int(len(age60files)/10)):
    age60temp = random.choice(age60files)
    age60files.remove(age60temp)
    age60test.append(age60temp)

for j in range(int(len(age61files)/10)):
    age61temp = random.choice(age61files)
    age61files.remove(age61temp)
    age61test.append(age61temp)

for j in range(int(len(age62files)/10)):
    age62temp = random.choice(age62files)
    age62files.remove(age62temp)
    age62test.append(age62temp)

for j in range(int(len(age63files)/10)):
    age63temp = random.choice(age63files)
    age63files.remove(age63temp)
    age63test.append(age63temp)

for j in range(int(len(age64files)/10)):
    age64temp = random.choice(age64files)
    age64files.remove(age64temp)
    age64test.append(age64temp)


for j in range(int(len(age65files)/10)):
    age65temp = random.choice(age65files)
    age65files.remove(age65temp)
    age65test.append(age65temp)

for j in range(int(len(age66files)/10)):
    age66temp = random.choice(age66files)
    age66files.remove(age66temp)
    age66test.append(age66temp)

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
