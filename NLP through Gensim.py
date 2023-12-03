#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


pip install gensim


# In[1]:


import gensim
import os


# # Gensim - it is an open source library in python written by Radim Rehurek which is used in unsupervised topic modeling and natural language processing.
# It is designed to extract semantic topics from documents. It can handle large text collections

# In[2]:


from nltk import sent_tokenize
from gensim.utils import simple_preprocess

story = []
for filename in os.listdir("C:\\Users\\gajendra singh\\OneDrive\\Desktop\\GamesOfThrone"):
    f = open(os.path.join("C:\\Users\\gajendra singh\\OneDrive\\Desktop\\GamesOfThrone", filename))
    corpus = f.read()
    raw_sent = sent_tokenize(corpus)
    for sent in raw_sent:
        story.append(simple_preprocess(sent))


# In[3]:


len(story)


# In[4]:


story


# In[5]:


model = gensim.models.Word2Vec(
window = 10,
min_count = 2
)


# In[6]:


model.build_vocab(story)


# In[8]:


model.train(story, total_examples = model.corpus_count, epochs = model.epochs)


# In[9]:


model.wv.most_similar('daenerys')


# In[10]:


model.wv.most_similar('daenerys')


# In[11]:


model.wv.doesnt_match(['jon', 'rikon', 'arya', 'sansa', 'bran'])


# In[12]:


model.wv.doesnt_match(['cersei', 'jaime', 'bronn', 'tyrion'])


# In[13]:


model.wv['jon']


# In[14]:


model.wv['king'].shape


# In[16]:


model.wv.similarity('arya','sansa')


# In[17]:


model.wv.similarity('tywin','sansa')


# In[18]:


model.wv.get_normed_vectors().shape


# In[19]:


y = model.wv.index_to_key


# In[20]:


from sklearn.decomposition import PCA


# In[21]:


pca = PCA(n_components = 3)


# In[23]:


x = pca.fit_transform(model.wv.get_normed_vectors())


# In[24]:


x[:5]


# In[25]:


import plotly.express as px
fig = px.scatter_3d(x[:500] , x=0, y=1, z=2, color=y[:500])
fig.show()


# In[ ]:




