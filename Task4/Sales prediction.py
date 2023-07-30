#!/usr/bin/env python
# coding: utf-8

# In[47]:


#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[48]:


#reading the csv file
data = pd.read_csv('advertising.csv')


# In[49]:


data.head()


# In[50]:


#information about the dataset
data.info()


# In[51]:


#describing the data
data.describe()


# In[52]:


#We check if there are any null values
data.isnull().sum()


# It's clear that there are no values.

# In[53]:


# Visualize the relationship between TV and sales using scatter plots
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.scatter(data['TV'], y)
plt.xlabel('TV')
plt.ylabel('Sales')

plt.subplot(1, 3, 2)
plt.scatter(data['Radio'], y)
plt.xlabel('Radio')
plt.ylabel('Sales')

plt.subplot(1, 3, 3)
plt.scatter(data['Newspaper'], y)
plt.xlabel('Newspaper')
plt.ylabel('Sales')

plt.tight_layout()
plt.show()


# In[54]:


# Step 3: Split the data into training and test sets
X = data.drop('Sales', axis=1)  # Input features (TV,Radio,Newspaper)
y = data['Sales']  # Target variable (sales amount)


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[56]:


# Step 4: Select and train a model
model = LinearRegression()
model.fit(X_train, y_train)


# In[57]:


# Step 5: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
acc=r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Accuracy:", acc)


# In[58]:


# Step 6: Make predictions
new_data = pd.DataFrame({
    'TV': [100, 200, 150],
    'Radio': [20, 30, 25],
    'Newspaper': [40, 50, 45]
})


# In[59]:


#predicted sales
predictions = model.predict(new_data)
print("Predicted sales:", predictions)

