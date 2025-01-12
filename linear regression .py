#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv("Salary_Data[1].csv")


# In[4]:


df.head(10)


# In[5]:


df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}) 
df['Education Level'] = df['Education Level'].map({'Bachelor': 0, 'Master': 1, 'PhD': 2})


# In[6]:


df = df.dropna()
#dropping the null values


# In[57]:


print(df['Education Level'].unique())
#Converting categorical data to numerical data
df['Education Level'] = df['Education Level'].map({'Bachelor': 0, 'Master': 1, 'PhD': 2})
print(df)


# In[58]:


X = df[['Age', 'Gender', 'Years of Experience']].values 
y = df['Salary'].values


# In[59]:


print(X)


# In[60]:


print(y)


# In[61]:


X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# In[62]:


X = np.c_[np.ones(X.shape[0]), X]


# In[63]:


W = np.zeros(X.shape[1])


# In[64]:


def hypothesis(X, W): 
    return np.dot(X, W)


# In[65]:


def cost_function(X, y, W): 
    m = len(y) 
    return (1 / (2 * m)) * np.sum((hypothesis(X, W) - y) ** 2)


# In[66]:


# Gradient Descent Algorithm
def gradient_descent(X, y, W, alpha, num_iterations):
    m = len(y)
    cost_history = []
    
    for i in range(num_iterations):
        # Updating the weights
        W = W - (alpha / m) * np.dot(X.T, (hypothesis(X, W) - y))
        
        # Calculating cost and store in history
        cost_history.append(cost_function(X, y, W))
        
    return W, cost_history


# In[67]:


alpha = 0.001 
num_iterations = 1000

# Training the model
W, cost_history = gradient_descent(X, y, W, alpha, num_iterations)

print("Weights:", W)
print("Cost history:", cost_history[-10:])  # last 10 cost values


# In[69]:


def predict(X, W): return hypothesis(X, W)


# In[70]:


predicted_salaries = predict(X, W)


# In[71]:


df['Predicted Salary'] = predicted_salaries


# In[72]:


print("DataFrame with Predicted Salaries:\n", df.head())


# In[74]:


df.head(10)


# In[ ]:




