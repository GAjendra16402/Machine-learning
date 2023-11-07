#!/usr/bin/env python
# coding: utf-8

# # Recurrent Neural Network and its applicationss =>

# # Recurrent Neural Networks (RNNs) are a type of neural network designed to process sequential data by maintaining an internal state, or memory, 
# to capture information from previous inputs. They are particularly useful when dealing with sequential and temporal data, 
# as they can learn patterns and dependencies over time.
# 
# Here are some key reasons why RNNs are used:
# 
# (1). Sequential Data Processing: RNNs excel at processing sequential data, where the order of elements matters, 
#      such as time series data, natural language processing (NLP), speech recognition, 
#      and handwriting recognition. They can capture dependencies between different elements in the sequence.
# 
# (2). Variable-Length Inputs: RNNs can handle variable-length inputs and produce corresponding outputs of the same sequence length. This flexibility 
#      is valuable in applications where inputs or outputs have varying lengths, such as text generation or speech synthesis.
# 
# (3). Memory and Contextual Information: RNNs maintain internal memory to store information about past inputs,
#      allowing them to retain context and information from earlier elements in the sequence. This memory enables 
#      the network to make decisions based on previous inputs and their relationships.
# 
# (4). Time Series Analysis: RNNs are commonly used for analyzing time series data, such as financial data, weather data, and physiological signals. 
#      By considering the temporal nature of the data, RNNs can model trends, dependencies, and patterns over time.
# 
# (5). Natural Language Processing (NLP): RNNs have proven to be highly effective in NLP tasks, including language modeling, machine translation, sentiment analysis, text classification, 
#      and named entity recognition. They can capture the semantic and syntactic structure of language by processing words or characters sequentially.

# # What is Sequential Data ?
# 
# Sequential data refers to a type of data where the order or sequence of elements carries significance and affects the interpretation or analysis of the data. In sequential data, the position or arrangement of elements conveys information or patterns that need to be captured and understood.
# 
# Real-life examples of sequential data include
# 
# (1). Time Series Data: Time series data is a classic example of sequential data. It involves a sequence of data points recorded over time, where each point represents a measurement or observation taken at a specific time. 
#      Examples include stock prices, temperature recordings, heart rate monitoring, and daily sales data. The order of the data points is crucial for understanding trends, seasonality, and patterns over time.
# 
# (2). Natural Language Text: Textual data, such as sentences or paragraphs in natural language, is inherently sequential. The order of words and characters
#      carries meaning and determines the semantics and syntax of the text. Language models, machine translation, sentiment analysis, and text generation tasks all
#      rely on capturing the sequential structure of text. (3). Music and Audio Signals: Musical compositions and audio signals are sequential in nature. 
#      Musical notes played over time form a sequence that needs to be captured to understand melodies, rhythms, and harmonies. Similarly, audio signals like speech,
#      music recordings, or environmental sounds can be represented as a sequence of samples over time.
# 
# (4). DNA Sequences: In genetics, DNA sequences represent the order of nucleotides (adenine, thymine, cytosine, and guanine) that make up an organism's 
#      genetic material, Analyzing and understanding DNA sequences is crucial in various biological applications, including genetic research, disease diagnosis, and evolutionary studies.
# 
# (5). User Behavior Data: Sequential data can also be found in user behavior logs, such as web clickstreams or transaction histories. 
#      The order of actions taken by users provides insights into their browsing patterns, preferences, or purchasing behaviors. Analyzing this sequential data can help optimize

# In[1]:


import numpy as np


# In[28]:


# define the RNN parameter
input_size = 4 # Dimensionalty of the input at each time setp
hidden_size = 3 # Dimensionalty of the hidden layer
output_size = 2 # Dimensionalty of the output at each time step


# In[29]:


# define the input sequence
sequence = np.array([[1,2,3,4],
                   [5,6,7,8],
                   [9,10,11,12]])


# In[30]:


# initialize the RNN weights

Wxh = np.random.randn(hidden_size, input_size)  # Weight matrix for input-to-hidden connections
Whh = np.random.randn(hidden_size, hidden_size)  # Weight matrix for hidden-to-hidden connections
Why = np.random.randn(output_size, hidden_size)  # Weight matrix for hidden-to-output connections
bh = np.zeros((hidden_size, 1))  # Bias for hidden state
by = np.zeros((output_size, 1))  # Bias for output


# In[31]:


h_prev = np.zeros((hidden_size, 1))


# In[34]:


for x in sequence:
    # convert input to column vector
    x = x.reshape(-1,1)
    
    # claculate the hidden state
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h_prev)+bh)
    
    y = np.dot(Why, h) + by


# In[39]:


print('output:', y.ravel())
h_prev = h


# In[ ]:


# Print the output at each time step
   print("Output:", y.ravel())
   
   # Update the hidden state for the next time step
    h_prev = h


# In[40]:


import tensorflow as tf


# In[41]:


# define the loss function
rnn = tf.keras.layers.SimpleRNN(units=64)


# In[46]:


loss_fn = tf.keras.losses.MeanSquaredError()


# In[47]:


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


# In[48]:


input_data = tf.random.normal(shape=(32,10,32))
target_data = tf.random.normal(shape=(32,64))


# In[49]:


with tf.GradientTape() as tape:
    predictions = rnn(input_data)
    loss_value = loss_fn(target_data, predictions)


# In[50]:


gradients = tape.gradient(loss_value, rnn.trainable_variables)


# In[51]:


optimizer.apply_gradients(zip(gradients, rnn.trainable_variables))


# In[52]:


gradients


# In[ ]:




