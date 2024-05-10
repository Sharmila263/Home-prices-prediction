#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n


# In[2]:


get_ipython().system('pip install word2number')


# In[4]:


df=pd.read_csv(r"C:\Users\sharm\Downloads\hiring.csv")


# In[5]:


df


# In[6]:


df.experience = df.experience.fillna("zero")
df


# In[8]:


import math
median_test_score = math.floor(df['test_score(out of 10)'].mean())
median_test_score


# In[11]:


df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(median_test_score)
df


# In[12]:


df.experience = df.experience.apply(w2n.word_to_num)
df


# In[13]:


model=linear_model.LinearRegression()


# In[16]:


model.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])


# In[17]:


model.predict([[2,9,6]])


# In[18]:


model.predict([[12,10,10]])

