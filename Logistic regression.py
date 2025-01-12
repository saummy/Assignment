#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[19]:


data=pd.read_csv("loan_data.csv")


# In[20]:


data.head(10)


# In[21]:


data["person_gender"] = data["person_gender"].map({"female": 0, "male": 1})


# In[22]:


data["person_home_ownership"] = data["person_home_ownership"].map({"RENT": 0, "OWN": 1, "MORTGAGE": 2})


# In[23]:


data["loan_intent"] = data["loan_intent"].astype("category").cat.codes


# In[24]:


data["previous_loan_defaults_on_file"] = data["previous_loan_defaults_on_file"].map({"No": 0, "Yes": 1})


# In[25]:


data["person_education"]=data["person_education"].map({"High School":0,"Bachelor":1,"Associate":2,"Master":3})


# In[26]:


data.head()


# In[38]:


X = data.drop(columns=["loan_status"]).values
y = data["loan_status"].values


# In[39]:


X = (X - X.mean(axis=0)) / X.std(axis=0)


# In[47]:


print(X)


# In[117]:


X = np.nan_to_num(X, nan=0.0)


# In[118]:


X = np.c_[np.ones(X.shape[0]), X]


# In[119]:


print(X)


# In[120]:


weights = np.random.rand(X.shape[1]) * 0.01


# In[121]:


print(weights)


# In[122]:


#Sigmoid ffunction
def sigmoid(z):
    z = np.clip(z, -500, 500)  
    return 1 / (1 + np.exp(-z))


# In[123]:


def compute_loss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


# In[124]:


def gradient_descent(X, y, weights, learning_rate, epochs):
    for epoch in range(epochs):
        # Predictions
        z = np.dot(X, weights)
        y_pred = sigmoid(z)  
        
        # gradient
        gradient = np.dot(X.T, (y_pred - y)) / y.size

        # Updating weights
        weights -= learning_rate * gradient

        # loss
        loss = compute_loss(y, y_pred)  

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss}")

    return weights


# In[125]:


print(weights)


# In[126]:


learning_rate = 0.01 
epochs = 1000 
weights = gradient_descent(X, y, weights, learning_rate, epochs)


# In[127]:


def predict(X, weights):
    return (sigmoid(np.dot(X, weights)) >= 0.5).astype(int)

y_pred = predict(X, weights)


# In[128]:


print(y_pred)


# In[129]:


accuracy = np.mean(y_pred == y) 
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[ ]:




