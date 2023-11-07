#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# # How I select which kernal I have use for which data in SVC algorithm

# # When selecting the kernel for the Support Vector Classifier(SVC) algorithm, you need to consider the characteristic of your data
# and the problem you are trying to solve. The choice of the kernel will determine the maping of the
# data to a higher-dimensional space where it can be linearly separable.
# 
# 
# (1). Linear Kernel:
#     The linear Kernel is the simplest kernel and is suitable for linearly seprable data.
#     
#     It works well when there is a clear linear boundry between classes.
#     
#     It is computationally efficient and often a good starting point.
#     
# In Scikit-learn, you can use Kernel = 'linear' to specify the linear kernel.
# 
# (2) polynomial Kernel:
#     
#     The polynomial kernel maps the data into a higher-dimensional space using polynomial functions.
#     
#     it is useful when the decision boundary is curved or has higher degrees of complexity.
#     
#     the degree of the polynbnomial, which determines the complexity, can be specified.
#     
# In Scikit-learn, you can use kernel = 'poly' and specify the degree with degree parameter.
# 
# (3). gausssian(RBF) Kernel:
#     
#     The Gaussian or Radial Basis Function(RBF) kernel maps the data into an infinite-dimensional space.
#     
#     it is suitable for the non-linearly seprable data and works well when the decision boundary is complex.
#     
#     it is popular choice due to its flexibility and ability to capture intricate patterns.
#     
#     the gamma parameter controls the smoothness of the decision boundary.
#     
# In scikit-learn , You can use kernel='RBf' and adjust gamma parameter.
# 
# (4) Sigimoid Kernel:
#     
#     The sigimoid kernel maps the data into a higher-dimensional space using the sigmoid function.
#     
#     it is useful when the decision boundary is S-shaped or logistic in nature.
#     
#     it can be effictive in binary classification problems.
#     
#     the gamma and coef0 parameter controls 

# # EXAMPLE-1

# In[4]:


from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[6]:


X,y = make_classification(n_samples = 100, n_features=2, n_informative=2, 
                         n_redundant=0, random_state=42)

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# create An SVM classifier with a linear kernel
svc = svm.SVC(kernel='linear')

# train the classifier on the training data
svc.fit(X_train, y_train)

# make prediction on the test data
predictions = svc.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, predictions)
print('Accuarcy', accuracy)


# # EXAMPLE-2

# In[15]:


import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn.svm import SVC


# In[16]:


# load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2] # consider only the first two features : sepal length  and width
y = iris.target


# In[17]:


# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# In[18]:


svc = SVC(kernel='linear')


# In[21]:


svc.fit(X_train, y_train)

# prediction on the test data
predictions = svc.predict(X_test)

# Evaluate the accuarcy of the classifier
accuracy = accuracy_score(y_test, predictions)
print('Accuracy:', accuracy)


# Plot the decision boundary
X_min, X_max = X[:, 0].min()- 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min()- 0.5, X[:, 1].max() +0.5

X,y = np.meshgrid(np.arange(X_min, X_max, 0.02),
                 np.arange(y_min, y_max, 0.02))


# In[23]:


Z= svc.predict(np.c_[X.ravel(), y.ravel()])
Z= Z.reshape(X.shape)

plt.contourf(X,y,Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot the trainig points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k')
# plot test points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, marker='x')


plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM Decision Boundary for Iris Dataset')
plt.show()


# In[ ]:




