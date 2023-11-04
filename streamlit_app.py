#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


d1=pd.read_csv('train (5).csv')


# In[3]:


d2=pd.read_csv('test (2).csv')


# In[4]:


submission=pd.read_csv('gender_submission.csv')


# In[5]:


d1.head()


# In[6]:


d2.head()


# In[7]:


d1.describe()


# In[8]:


fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(d1.corr(),annot=True,cmap='Blues')


# In[9]:


d1=d1.drop(['Cabin'],axis=1)


# In[10]:


d1=d1.drop(['Name'],axis=1)


# In[11]:


d1.isnull().sum()


# In[12]:


d1.size


# In[13]:


d2.size


# In[14]:


d1.shape


# In[15]:


d2.shape


# In[16]:


d1.dtypes


# In[17]:


d1.Sex=d1.Sex.astype('category').cat.codes


# In[18]:


d1.Ticket=d1.Ticket.astype('category').cat.codes


# In[19]:


d1.Embarked=d1.Embarked.astype('category').cat.codes


# In[20]:


d1.dtypes


# In[21]:


d1.head()


# In[22]:


d2=d2.drop(['Cabin'],axis=1)


# In[23]:


d2=d2.drop(['Name'],axis=1)


# In[24]:


d2.Sex=d1.Sex.astype('category').cat.codes


# In[25]:


d2.Ticket=d1.Ticket.astype('category').cat.codes


# In[26]:


d2.Embarked=d1.Embarked.astype('category').cat.codes


# In[27]:


d2.dtypes


# In[28]:


d1.Age=d1.Age.fillna(d1.Age.median())


# In[29]:


d1.isnull().sum()


# In[30]:


d2.Age=d2.Age.fillna(d2.Age.median())


# In[31]:


d2.isnull().sum()


# In[32]:


d2.Fare=d2.Fare.fillna(d2.Fare.median())


# In[33]:


d2.isnull().sum()


# In[34]:


sns.pairplot(d1)


# In[35]:


d2=d2.drop(['PassengerId'],axis=1)


# In[36]:


d1=d1.drop(['PassengerId'],axis=1)


# In[37]:


X=d1.drop(['Survived'],axis=1)
y=d1['Survived']


# In[38]:


X.columns


# In[39]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
col_to_standard=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
       'Embarked']
X[col_to_standard]=scaler.fit_transform(X[col_to_standard])


# In[40]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
col_to_standard=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
       'Embarked']
d2[col_to_standard]=scaler.fit_transform(d2[col_to_standard])


# In[41]:


from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2)


# In[42]:


from sklearn.neighbors import KNeighborsClassifier


# In[43]:


model=KNeighborsClassifier()


# In[44]:


model.fit(X_train,y_train)


# In[45]:


preds=model.predict(X_train)


# In[46]:


len(preds)


# In[47]:


from sklearn.metrics import classification_report,f1_score,accuracy_score


# In[48]:


print(classification_report(y_train,preds))


# In[49]:


y_pred_valid = model.predict(X_valid)


# In[50]:


print("Train Accuracy:",accuracy_score(y_train,preds))
print("valid Accuracy:",accuracy_score(y_valid,y_pred_valid))


# In[51]:


preds_test=model.predict(d2)


# In[52]:


preds_test


# In[53]:


import csv
submission['Survived'] = preds_test
submission.to_csv('KNN_SUBMISSION.csv', index=False)
submission = pd.read_csv('KNN_SUBMISSION.csv')
submission

#saving the  model
import pickle
pickle.dump(clf,open('titanic_clf.pkl','wb'))
