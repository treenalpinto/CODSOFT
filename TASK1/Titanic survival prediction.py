#!/usr/bin/env python
# coding: utf-8

# In[33]:


#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[34]:


#reading the dataset
df=pd.read_csv('tested.csv')


# In[35]:


df.head()


# In[36]:


print("number of passengers :",len(df.index))


# In[37]:


#this plot shows how many survived and how many did not
#0 represents they didnt survive and 1 represents they survived
sns.countplot(x = 'Survived', data = df)


# In[79]:


#Here,PassengerId:passenger ID
#Survived Survival: (0 = No; 1 = Yes)
#Pclass Passenger Class: (1 = 1st; 2 = 2nd; 3 = 3rd)
#Name :Name
#Sex :Sex
#Age :Age
#SibSp: Number of Siblings/Spouses Aboard
#Parch :Number of Parents/Children Aboard
#Ticket :Ticket Number
#Fare :Passenger Fare
#Cabin :Cabin
#Embarked Port of Embarkation: (C = Cherbourg; Q = Queenstown; S = Southampton)


# In[38]:


#count of people who survived and who didnt
count_survived = df["Survived"].value_counts()
count_survived


# In[39]:


# To know how many male and female survived.
sns.countplot(x = 'Survived', hue = 'Sex', data = df)


# This above plot shows that majority of males did not survive ,which means that average women were three times more likely to survive than the man.

# In[40]:


sns.countplot(x = 'Survived', hue = 'Pclass', data = df)


# In the above plot we can see that passenger who did not survive were from class three.

# # Now lets see the age distribution of the passengers

# In[41]:


df['Age'].plot.hist()


# In the above plot , we can see that the number of young aged people are more.

# In[42]:


df.info()


# In[43]:


sns.countplot(x = 'SibSp', data = df)


# The above plot tells us that majority of the passengers were not travelling with their siblings and spouse.

# In[44]:


#Checking the null values
df.isnull().sum()


# We can see that , there are null values in Age,Fare and Cabin columns. We can plot it as

# In[45]:


sns.heatmap(df.isnull(), yticklabels= False)


# In[46]:


# checking for outliers
sns.boxplot(x ="Pclass", y="Age",data = df)


# Here , we can see that passenegers in class 1 tend to be aged than the passengers in class 3

# In[47]:


# Droping the feature
df.drop('Cabin',axis= 1, inplace = True)


# In[48]:


df.head(5)


# In[49]:


# Dropping the na values
df.dropna(inplace = True)


# In[50]:


df.info()


# In[51]:


sns.heatmap(df.isnull())


# This above plot indicates that there are no null values present.

# In[52]:


# encoding
sex = pd.get_dummies(df['Sex'],drop_first=True).astype('int')
sex.head(5)


# In[53]:


embark = pd.get_dummies(df['Embarked'], drop_first = True).astype('int')
embark.head()


# In[54]:


Pcl  = pd.get_dummies(df['Pclass'], drop_first = True).astype('int')
Pcl.head()


# In[55]:


# joining sex,embark,Pcl with the original data set
titanic_data = pd.concat([df,sex, embark,Pcl],axis = 1)


# In[56]:


titanic_data


# In[57]:


# Droping the Columns
titanic_data.drop(['Sex','Embarked','Name','PassengerId','Pclass','Ticket'],axis= 1, inplace = True)


# In[58]:


titanic_data.head()


# In[59]:


#Train Data
x = titanic_data.drop("Survived", axis= 1)
y = titanic_data['Survived']


# In[60]:


x.columns=x.columns.astype(str)


# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# In[63]:


from sklearn.linear_model import LogisticRegression


# In[64]:


logmodel=LogisticRegression()


# In[65]:


logmodel.fit(X_train, y_train)


# In[68]:


predictions = logmodel.predict(X_test)


# In[69]:


from sklearn.metrics import classification_report


# In[71]:


print(classification_report(y_test, predictions))


# In[74]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[78]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,predictions)
print("accuracy of the model:",acc)

