#!/usr/bin/env python
# coding: utf-8

# Working on dataset of SampleTweets.

# ### import the necessary library

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


from sklearn.metrics import confusion_matrix


# ### load the dataset

# In[3]:


dataset = pd.read_csv('SampleTweets.csv',header = None )


# In[4]:


#check the shape of dataset
print(dataset.shape)
#print first 3 rows of datase
dataset.head(3)


# In[5]:


# drop column 2 and last
dataset.drop(columns = [1,3], inplace = True)


# In[6]:


dataset.head(3)


# In[7]:


# unique classes of the tweets
dataset[0].unique()


# In[8]:


# creating a new column to store the classes in the form of integers
dataset['y'] = 0


# In[9]:


for i in range(dataset.shape[0]):
    
    if dataset.iloc[i,0] == '|positive|':
        dataset.iloc[i,2] = 1
        
    elif dataset.iloc[i,0] == '|negative|':
        dataset.iloc[i,2] = -1


# In[10]:


# droping the first column
dataset.drop(columns = [0], inplace  = True)


# In[11]:


dataset.head(3)


# In[12]:


print(dataset['y'].unique())


# ### Natural Language Processing

# In[13]:


import re
import nltk


# In[14]:


from nltk.corpus import stopwords


# In[15]:


from nltk.stem.porter import PorterStemmer


# In[16]:


corpus = []
# filtering out the irrelevant stopwords from the tweets and storing it in a list
for i in range(0, dataset.shape[0]):
    tweet = re.sub('[^a-zA-Z]', ' ', dataset[2][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)


# In[17]:


# print first three list items
corpus[:3]


# ### Creating the X and y dataset to train

# taking the top 2000 words occuring in the dataset and creating a bag of words which will use to represent each
# row in the corpus individually        

# In[18]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)


# In[19]:


X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# In[20]:


print(X.shape)
print(y.shape)


# In[21]:


# Splitting the dataset in training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 32)


# # Naive Bayes

# In[22]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[23]:


y_pred = classifier.predict(X_test)


# In[24]:


cm = confusion_matrix(y_test,y_pred)
print(cm)
print("Accuracy of Naive Bayes>>",(cm[0,0] + cm[1,1] + cm[2,2])/ X_test.shape[0])


# # Logistic Regression

# In[25]:


from sklearn.linear_model import LogisticRegression
lg_classifier = LogisticRegression(random_state = 12, max_iter = 1000, solver = 'saga', multi_class = 'multinomial', warm_start = True, n_jobs = -1)


# In[26]:


lg_classifier.fit(X_train, y_train)


# In[27]:


y_pred = lg_classifier.predict(X_test)


# In[28]:


cm = confusion_matrix(y_test,y_pred)
print(cm)
print("accuracy of logistic regression>>",(cm[0,0] + cm[1,1] + cm[2,2])/ X_test.shape[0])


# # K-Nearest Neighbors

# In[29]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski', n_jobs = -1)


# In[30]:


knn.fit(X_train,y_train)


# In[31]:


y_pred = knn.predict(X_test)


# In[32]:


cm = confusion_matrix(y_test,y_pred)
print(cm)
print("accuracy of K-Nearest Neighbors>>",(cm[0,0] + cm[1,1] + cm[2,2])/ X_test.shape[0])


# # SVM (Support Vector Machine)

# In[33]:


from sklearn.svm import SVC
svc_classifier = SVC(kernel = 'linear', random_state = 12, coef0 = 1, gamma = 'auto')


# In[34]:


svc_classifier.fit(X_train,y_train)


# In[35]:


y_pred = svc_classifier.predict(X_test)


# In[36]:


cm = confusion_matrix(y_test,y_pred)
print(cm)
print("accuracy of SVM with linear kernel>>",(cm[0,0] + cm[1,1] + cm[2,2])/ X_test.shape[0])


# # Decision Tree Classification

# In[37]:


from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state  = 32)


# In[38]:


dt_classifier.fit(X_train,y_train)


# In[39]:


y_pred = dt_classifier.predict(X_test)


# In[40]:


cm = confusion_matrix(y_test,y_pred)
print(cm)
print("accuracy of decision tree classification>>",(cm[0,0] + cm[1,1] + cm[2,2])/ X_test.shape[0])


# # Random Forest Classification

# In[41]:


from sklearn.ensemble import RandomForestClassifier
rdt_classifier = RandomForestClassifier(n_estimators=100, criterion='entropy',
                                   n_jobs = -1, random_state = 32)


# In[42]:


rdt_classifier.fit(X_train,y_train)


# In[43]:


y_pred = rdt_classifier.predict(X_test)


# In[44]:


cm = confusion_matrix(y_test,y_pred)
print(cm)
print("accuracy of random forest classification>>",(cm[0,0] + cm[1,1] + cm[2,2])/ X_test.shape[0])


# In[ ]:





# In[ ]:




