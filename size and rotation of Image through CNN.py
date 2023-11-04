#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras

from keras. preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img , img_to_array


# In[2]:


datagen = ImageDataGenerator(
rotation_range = 40,
width_shift_range = 0.2,
height_shift_range = 0.2,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True,
fill_mode = 'nearest')


# In[4]:


img = load_img("C:\\Users\\gajendra singh\\OneDrive\\Desktop\\gajendra\\logo 2.png")


# In[5]:


x = img_to_array(img)


# In[6]:


x = x.reshape((1,) + x.shape)


# In[7]:


i = 0 
for batch in datagen.flow(x, batch_size = 1, save_to_dir = "C:\\Users\\gajendra singh\\OneDrive\\Desktop\\gajendra\\image rotation" , save_prefix = 'logo', save_format = 'jpeg'):
    
    i += 1
    if i >10:
        break


# In[ ]:




