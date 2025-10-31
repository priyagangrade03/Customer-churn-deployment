#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mysql.connector
import warnings
warnings.filterwarnings("ignore")


# In[9]:


# connect to MYSQL database

# conn = mysql.connector.connect(
   # host = "localhost",
   # user = "root",
   # password = "12345",
   # database= "churn_db"
#)


# In[22]:


# query = "SELECT * FROM customers;"
#df = pd.read_sql(query, conn)
#df.info()
df = pd.read_csv("Customer CHurn Analysis.csv")
df.info()


# In[23]:


df.head()


# In[24]:


# Data Cleaning
df["gender"] = df["gender"].map({"Male":0, "Female":1})


# In[25]:


# EDA


# In[26]:


# Model Building

X = df.drop(columns=["churn"])
y = df["churn"]


# In[28]:


# feature engineering (scaling)

from sklearn.preprocessing import StandardScaler


# In[30]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[32]:


# train-test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)


# In[33]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)


# In[35]:


# model prediction
y_pred = model.predict(X_test)

#model evaluation

from sklearn.metrics import classification_report, confusion_matrix

cr = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


# In[39]:


print(f"classification report:\n{cr}\nconfusion matrix:\n{cm}")


# In[41]:


# save model
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("Model is saved")


# In[ ]:




