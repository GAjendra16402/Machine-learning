#!/usr/bin/env python
# coding: utf-8

# # What is glove ? and why we use it?

# GloVe, which stands for global vectors for word representation, is a popular method used in natural
# Language Processing(NLP) to learn word embeddings. Word embeddings are dense vector representation of words 
# in a contniuos vector space. These embeddings are designed to capture semantic and contextual information about words, which can
# be benificial for various NLP tasks.

# The GloVe algorithm was developed bt resarchers at Standford University and Introduced 
# in the paper"GloVe: Global Vector for Word Representation" by Pennington, socher, and Manning in 2014.
# It builds on the idea that word co-occurence statistic can provide meaningful insights into the relationship between the words.

# (1). Corpus Statistics Collection: GloVe first builds a word co-occurence matrix from a large corpus of text.
#     The matrix represents how often words co-occur within a certain context window(e.g. how often words appear together in a sentence or paragraph).

# (2). Word Embedding Learning: GloVe uses a factorization technique to factorize the word co-occurrence matrix into two lower-
# 
#     how often words co-occur within a certain context window (e.g., how often words appear together in a sentence or paragraph). 
#     dimensional matrices, one for words and another for their contexts. The row vectors of the word matrix and column vectors of the context matrix serve as the
# 
# (3). Training Objective: The word embeddings.
# 
#     Why do we use Glove?
# 
# (1). Meaningful Word Representations: GloVe produces word embeddings that carry semantic and contextual information, enabling the model to understand the meaning of words and thier relationship to each other. 
#      these embedding are offten useful in various NLP tasks, such as sentimental analysis, machine translation, text classification and word analogy tasks.
# 
# (2) Dimensionally Reduction: GLove reduces the dimensionally of the word representation comapared to one - hot encoded vectors , which are extermely high -dimensonal and space. 
#     this reduction in dimensionally makes the word embedding more computationally efficient and ,anageable.
# 
# (3) Improved performance: Models that leverage pre-trained Glove embedding tend to perform better on downstream NLP tasks, 
#     especially when the dataset used for pre-training is large and diverse.
# ```
# 
# 

# In[ ]:




