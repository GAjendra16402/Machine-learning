#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd


# In[52]:


df = pd.read_csv("C:\\Users\\gajendra singh\\OneDrive\\Desktop\\pandas\\click.csv")


# In[53]:


df.head()


# In[54]:


import matplotlib.pyplot as plt


# In[55]:


import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# In[56]:


df = df.drop(columns=['Timestamp'], axis = 1)


# In[57]:


df


# In[58]:


x = df.drop(columns = ['Clicked on Ad'],axis =1)
y = df['Clicked on Ad']


# In[59]:


lb = LabelEncoder()
df['City'] = lb.fit_transform(df['City'])
df['Gender'] = lb.fit_transform(df['Gender'])


# In[60]:


x['City'] = lb.fit_transform(x['City'])
x['Gender'] = lb.fit_transform(x['Gender'])
x['Country'] = lb.fit_transform(x['Country'])
x['Ad Topic Line'] = lb.fit_transform(x['Ad Topic Line'])


# In[61]:


x_train , x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=42)


# In[62]:


sc = StandardScaler()


# In[63]:


x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)


# In[64]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation = 'relu', input_dim = x_train.shape[1]),  # input layer
    tf.keras.layers.Dense(64, activation = 'relu'), # hidden layer
    tf.keras.layers.Dense(1 , activation = 'sigmoid') #output layer
])


# In[65]:


# define the loss function and metrics
loss_fn = tf.keras.losses.BinaryCrossentropy()
metrics = ['accuracy']


# In[66]:


# define the optimizer
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)


# In[67]:


model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)


# In[68]:


epochs = 10
batch_size = 32
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs, validation_split=0.2)


# In[70]:


test_loss, test_accuracy, = model.evaluate(x_test,y_test)
print('Test Loss:', test_loss)
print('Test Accuarcy:', test_accuracy)


# In[71]:


df1 = pd.read_csv("C:\\Users\\gajendra singh\\OneDrive\\Desktop\\pandas\\Attrition.csv")


# In[72]:


df1.head()


# In[73]:


df1.info()


# In[76]:


x = df1.drop(columns = ['Attrition'], axis=1)
y = df1['Attrition']


# In[77]:


categorical_features = ['BusinessTravel',
                       'Department',
                       'EducationField',
                       'Gender',
                       'JobRole',
                       'MaritalStatus',
                       'Over18',
                       'OverTime']


# In[78]:


x_encoded = pd.get_dummies(x, columns = categorical_features, drop_first=True)


# In[79]:


y_encoded = lb.fit_transform(y)
y_encoded


# In[80]:


x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, test_size = 0.2, random_state = 42)


# In[81]:


x_train


# In[82]:


sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)


# In[83]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units= 128, activation = 'relu', input_dim = x_train_scaled.shape[1]),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units = 64, activation = 'relu'), # hidden layer
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units = 1, activation = 'sigmoid') # output layer
])


# In[84]:


loss_fn = tf.keras.losses.BinaryCrossentropy()
metrics = ['accuracy']


# In[86]:


learning_rate = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


# In[87]:


model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)


# In[88]:


epochs = 50
batch_size = 8
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs, validation_split=0.2)


# In[89]:


test_loss, test_acc, = model.evaluate(x_test,y_test)
print('Test Loss:', test_loss)
print('Test Accuarcy:', test_acc)


# In[ ]:




