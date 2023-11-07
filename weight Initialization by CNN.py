#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df  = pd.read_csv("C:\\Users\\gajendra singh\\OneDrive\\Desktop\\pandas\\ushape.csv")


# In[3]:


df.head()


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


plt.scatter(df['X'], df['Y'], c = df['class'])


# In[32]:


x = df.iloc[: , 0:2].values
y = df.iloc[:, -1].values


# In[9]:


import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense


# In[10]:


model = Sequential()

model.add(Dense(2, activation = 'relu', input_dim = 2))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()


# In[11]:


# set parameter to 0
model.get_weights()


# In[12]:


initial_weights = model.get_weights()


# In[14]:


initial_weights[0] = np.zeros(model.get_weights()[0].shape)
initial_weights[1] = np.zeros(model.get_weights()[1].shape)
initial_weights[2] = np.zeros(model.get_weights()[2].shape)
initial_weights[3] = np.zeros(model.get_weights()[3].shape)


# 

# In[15]:


model.set_weights(initial_weights)


# In[16]:


model.get_weights()


# In[17]:


model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])


# In[19]:


history = model.fit(x,y, epochs = 100, validation_split = 0.2)


# In[20]:


model.get_weights()


# In[23]:


get_ipython().system('pip install mlxtend')


# In[24]:


from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x,y.astype('int') , clf = model , legend = 2)


# In[25]:


model = Sequential()

model.add(Dense(2, activation = 'tanh', input_dim = 2))
model.add(Dense(1, activation = 'sigmoid'))


# In[26]:


model.summary()


# In[27]:


model.get_weights()


# In[28]:


initial_weights = model.get_weights()


# In[29]:


model.get_weights()


# In[30]:


model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])


# In[31]:


history = model.fit(x,y, epochs = 100, validation_split = 0.2)


# In[ ]:




