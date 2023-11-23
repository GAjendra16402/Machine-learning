#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Regular Expression


# In[3]:


import re


# In[12]:


# 8 zero or one occurance 
# search for a sequence that starts with "he", followed by 1 or more (any) character, and am "o"
a =  "heelo world"

x = re.findall("he.?o", a)
x


# In[7]:


# 9. {} exactly the specified number of occurances.
a =  "heelo world"
# search for a sequence that starts with "he", followed exactly 2 (any) character, and an 'o'
x = re.findall("he.{2}o", a)
x


# In[10]:


# 10. | either or
txt = "the rain in spain mainly falls in the plain"

x = re.findall("falls|stays", txt)
print(x)


# # Special Sequence

# In[13]:


# a special sequnece is a \ followed by one of the character and has a special meaning.
#1. /A => returns a match if the specified cahracter are at the begining of the string


# In[18]:


txt = "the rain in  spain the"
x = re.findall("^the", txt)
x


# In[19]:


x = re.findall("\Athe",txt)
x


# In[20]:


# 2. \b ==> returns a match where specified character are at beginig or at the end of a word

txt= "the rain in spain main cain train"
x = re.findall(r"ain\b", txt)
x


# In[21]:


# 3 \B==> Returns a match where a specified character present, but not at the begining (or the end) of a word.
# check if "ain" is present, biut NOT at the begining of a word.


# In[22]:


# the "r" in the beginig is making sure that the string is being treated as a 'raw string'

# check if "ain" is present, but not at the end of a word

x = re.findall(r"ain\B", txt)
x
if x:
    print("yes")
else:
    print("no")


# In[23]:


# 4. \d returns a match where the string contains digit (number from 0-9)

txt = "the rain in 78 spain"
x = re.findall("\d", txt)
x


# In[24]:


# 5 \D returns a match wher the string doesnt contain the digit
# returns a match at every no-digit charater 
x = re.findall("\D", txt)
x


# In[25]:


# 6 \s returns a match where the string contains a white space chatacter
x = re.findall('\S', txt)
x


# In[26]:


# \w ==> return a match where the string conatins any word chracter (character from a to z, digits from 0-9, and the underscore character)


# In[27]:


x = re.findall('\w', txt)
print(x , end="")


# In[28]:


# 7. \W==> returns a match where the string does not contain any word character

x = re.findall("\W", txt)
print(x, end="")


# In[29]:


# \Z returns a match 


# # sets
# A set is a set of character inside a pair of square brackets[] with special meaning.

# In[30]:


#1. [arn] ==> return a match where one of the specified character (a,r or n) is present
txt = "I am in jaipur"
x = re.findall("[ampur]", txt)
x


# In[31]:


# 2. [a-n]==> returns a match for any lower case character, alphabatecly between a to n
x = re.findall("[a-z]", txt)
x


# In[32]:


#3. [^arn]==> returns a match for character except a,r,n

x = re.findall("[^arn]", txt)
x


# In[33]:


#4. [01234] = returns a match where any of the specified digit (0,1,2,3) are present
txt = "I am 1 in 13 jaipur"
x = re.findall('[0123]', txt)
x


# In[34]:


#5 . [0-9] == reurns a match for any digit between 0-9

txt = "I am 354 in jaipur 243"
x = re.findall('[0-9]', txt)
x


# In[43]:


# 6.[0-5][0-9] == returns a match for any two digit number from 00 amd 99

txt = "mai times before 11:40 AM"
# check if the string has any two-digit numbers, from 00 to 99
x = re.findall('[0-5][0-9]', txt)
x


# In[44]:


#7. [a-zA-Z] == find all the character alphabatically btween a to z , lower case and upper case
x = re.findall('[a-zA-Z]', txt)
x


# In[45]:


# 8. [+] in sets,+, *,.,|,(),$,{} has no special meaning, so[+] means: return a matc for any + character in the string

txt = "8 times before 11+45 AM"
x = re.findall("[+]", txt)
x


# # The findall() function
# this funcrtion returns a list of all containig matches
# 

# In[47]:


txt = "the rain in spain"
x = re.findall("ai", txt)
x


# # Match Object
# A match object is an object containing information about the search and the result
# 
# Note: If there is no match, the value NOne will be returned, instead of the match object
# 

# In[48]:


txt = "the rain in spain"
x = re.search("ai",txt)
x


# # the match object hhas properties and methods used to retrieve information about the search, ant the result.

# In[ ]:


# spam() returns a tuple containig the start-, and end postion of the match.
# .string returns the string passed into the function.
# group() returns the part of the string where there was a match

