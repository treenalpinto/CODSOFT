#!/usr/bin/env python
# coding: utf-8

# In[34]:


#Importing the libraries
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


# In[35]:


#read the dataset
df = pd.read_csv("creditcard.csv")


# In[36]:


df.head()


# In[37]:


#displaying the information about the dataset
df.info()


# In[38]:


#describing the data in the dataset
df.describe()


# In[39]:


#checking if there are any null values in the dataset
df.isnull().sum()


# Here, we can see that there are no null values.

# In[40]:


#splitting the data as legit and fraud where 0 means legit and 1 means fraud
legit = df[df['Class']==0]
fraud = df[df['Class']==1]
print(legit.shape)
print(fraud.shape)


# In[41]:


#here,we undersample the data 
legit_sample = legit.sample(n=492)
legit_sample.shape


# In[42]:


new_df = pd.concat([legit_sample,fraud])
new_df['Class'].value_counts()


# In[43]:


#here we take the independent and dependent variable
x = new_df.drop('Class', axis=1)
y = new_df['Class']


# In[44]:


#here,we split the data into test and train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[45]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(x_train, y_train)
predictions = logmodel.predict(x_test)


# In[46]:


from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(predictions, y_test))
print(classification_report(predictions, y_test))


# In[47]:


#Oversample
fraud_sample = fraud.sample(n=284315,replace=True)
fraud_sample.shape


# In[48]:


new_df = pd.concat([fraud_sample,legit])
new_df['Class'].value_counts()


# In[49]:


x = new_df.drop('Class', axis=1)
y = new_df['Class']


# In[50]:


#splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[51]:


#training the model '
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(x_train, y_train)
predictions = logmodel.predict(x_test)


# In[52]:


#finding the accuracy score and classification report
from sklearn.metrics import classification_report,accuracy_score
print(accuracy_score(predictions, y_test))
print(classification_report(y_test, predictions))


# In[53]:


#SMOTE oversampling
x = df.drop('Class', axis=1)
y = df['Class']


# In[54]:


from imblearn.over_sampling import SMOTE
sm = SMOTE()
x, y = sm.fit_resample(x, y)


# In[57]:


#splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[58]:


#training the data
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(x_train, y_train)
predictions = logmodel.predict(x_test)


# In[59]:


#finding the accuracy score
print(accuracy_score(predictions, y_test))
print(classification_report(predictions, y_test))

