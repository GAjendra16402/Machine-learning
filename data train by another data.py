#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[45]:


df = pd.read_csv("C:\\Users\\gajendra singh\\OneDrive\\Desktop\\pandas\\Indian_cities.csv")


# In[46]:


df.head()


# In[4]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.compose import ColumnTransformer


# In[5]:


df.info()


# In[12]:


df = df.drop(columns=['state_name','location'],axis=1)


# In[13]:


df.info()


# In[15]:


from sklearn.linear_model import Lasso
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score


# In[16]:


data = load_breast_cancer()
X,y = data.data , data.target


# In[23]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge


# In[24]:


x,y = data.data , data.target


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


from sklearn.metrics import mean_squared_error


# In[27]:


from sklearn.model_selection import cross_val_score


# In[28]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state = 42)


# In[29]:


sc = StandardScaler()


# In[30]:


x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)


# In[31]:


lasso = Lasso(alpha=0.1)
lasso.fit(x_train,y_train)


# In[32]:


y_pred = lasso.predict(x_test)


# In[33]:


y_pred = [1 if pred>=0.5 else 0 for pred in y_pred]


# In[34]:


accuracy = accuracy_score(y_test,y_pred)


# In[35]:


print('Accuracy :',accuracy)


# In[36]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[38]:


print(df.head())
print(df.info())


# In[39]:


X = df[['population_total', 'population_male', 'population_female']]  # Add other relevant features
y = df['total_graduates']


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[41]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[42]:


y_pred = model.predict(X_test)


# In[43]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[ ]:





# In[48]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.compose import ColumnTransformer


# In[ ]:




