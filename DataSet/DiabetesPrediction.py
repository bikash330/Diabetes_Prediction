#!/usr/bin/env python
# coding: utf-8

# # DIABETES PATIENTS

# We are using the dataset from National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether a patient has diabetes based on certain diagnostic measurements included in the dataset.

# Importing Required Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importing Important Libraries For Prediction

# Train Test Split: Technique for splitting data into training and testing sets to assess model performance.

# Logistic Regression: Method for predicting the probability of a binary 
# outcome using the logistic function.

# Accuracy: Metric measuring the proportion of correctly classified instances in a classification model.

# Sklearn: Python's Scikit-learn, a powerful machine learning library providing tools for data analysis and model building.

# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


#Loading the dataset
ds = pd.read_csv(r'C:\Users\Bikash shah\Downloads\MeriSkill\Project_Second\diabetes.csv')


# In[4]:


ds.head()


# In[5]:


ds.shape


# In[6]:


#looking if data type is correct
ds.dtypes


# In[7]:


ds.describe()


# In[8]:


#looking if there is any null/missing value 
sns.heatmap(ds.isnull())


# In[9]:


#Calculating correlation between each and every data (Correlation Matrix)
correlation = ds.corr()
print(correlation)


# In[10]:


#Visualizing the correlation
sns.heatmap(correlation)


# Training the model with Logistic Regression

# In[11]:


X = ds.drop("Outcome",axis=1) #independent variable
Y = ds["Outcome"] #dependent variable
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2) #train-test split with 0.2 test size


# In[12]:


model = LogisticRegression()


# In[13]:


model.fit(X_train,Y_train) #fitting train data of X with Y


# Making Prediction

# In[14]:


prediction = model.predict(X_test)


# In[15]:


print(prediction)


# In[16]:


accuracy = accuracy_score(prediction,Y_test)
print(accuracy)


# Hence, the accuracy of this model is 0.74.

# In[ ]:




