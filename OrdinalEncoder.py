#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv('tips.csv')


# In[4]:


df.head()


# In[5]:


df = df.drop(columns = ['total_bill' , 'tip' , 'size'])


# In[6]:


df.head(10)


# In[14]:


df['day'].value_counts()


# In[15]:


df['time'].value_counts()


# In[16]:


df['smoker'].value_counts()


# In[8]:


from sklearn.preprocessing import OrdinalEncoder


# In[17]:


oe = OrdinalEncoder(categories = [['Male' , 'Female'] , ['No' , 'Yes'] , ['Sat' , 'Sun' , 'Thur' , 'Fri'] , ['Dinner' , 'Lunch']])


# In[18]:


oe.fit(df)


# In[19]:


oe.transform(df)


# In[25]:


Df = pd.read_csv('cars.csv')


# In[26]:


Df


# In[27]:


Df['seller_type'].value_counts()


# In[28]:


Df['transmission'].value_counts()


# In[29]:


Df['owner'].value_counts()


# In[43]:


Df['fuel'].value_counts()


# In[38]:


Df = Df.drop(columns=['name'])


# In[39]:


Df


# In[40]:


from sklearn.preprocessing import OrdinalEncoder


# In[48]:


oe = OrdinalEncoder(categories = [['Petrol' , 'Diesel' , 'CNG' , 'LPG' , 'Electric'] , ['Trustmark Dealer', 'Dealer', 'Individual'] , ['Manual' , 'Automatic'] , ['First Owner' , 'Second Owner' , 'Third Owner' , 'Fourth & Above Owner' , 'Test Drive Car']])


# In[49]:


oe.fit(Df)


# In[50]:


oe.transform(Df)


# In[ ]:




