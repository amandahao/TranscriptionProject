#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import os
import urllib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[10]:


DATAFILE = '/Users/amandahao/Downloads/ohio.csv'
AGE_RECT = {'x': 45*5, 'y': 195*5,  'w': 60*5, 'h': 35*5}


# In[11]:


inputdata = pd.read_csv(DATAFILE, delimiter=',')
inputdata


# In[12]:


images = []
for imagelink in inputdata['na_url']:
    req = urllib.request.urlopen(imagelink)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    images.append(cv2.imdecode(arr, -1))
    print(len(images))
    if len(images) > 20: 
        break


# # Image Clean Up Strategy (Save this for later)
# 1. Rotate the image
# 2. Take out the boundary

# In[14]:


ageimage = images[17][AGE_RECT['y']:AGE_RECT['y']+AGE_RECT['h'], AGE_RECT['x']:AGE_RECT['x']+AGE_RECT['w']]
plt.imshow(ageimage, cmap=plt.cm.gray)


# In[15]:


plt.imshow(images[20], cmap=plt.cm.gray)


# In[16]:


inputdata['na_url'][13]


# In[ ]:




