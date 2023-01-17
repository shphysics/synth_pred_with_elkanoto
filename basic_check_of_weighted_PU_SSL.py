#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

import sklearn
import csv
import pandas as pd
import pymatgen as mg
import random
import os
from bisect import bisect_left   
from sklearn import preprocessing
from sklearn.datasets.base import Bunch
from sklearn import svm

#from sklearn import datasets, svm
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn import externals
import six
# import six
# import sys
# sys.modules['sklearn.externals.six'] = six

import sys
sys.path.insert(1,'/home/users/willcai1/anaconda3/envs/my_pymatgen/lib/python3.8/site-packages')
import pulearn
from pulearn import ElkanotoPuClassifier
from pulearn import WeightedElkanotoPuClassifier

style.use("ggplot")


# In[2]:


def format_dataset(X, y, size = -1):
    random.seed(a=5)
    length = np.size(y)
    indices = np.arange(length)
    np.random.shuffle(indices)
    new_X = []
    new_y = []
    for x in indices:
        new_X.append(X[x])
        new_y.append(y[x])
    if size == -1:
        return Bunch(data=new_X, target=new_y)
    return Bunch(data=new_X[0:size], target=new_y[0:size])


# In[3]:


X_train = np.array([[1,1,1], [1,2,3],[2,4,1],[2,2,3],[4,3,0],[9,10,11],[8,8,5],[6,7,9], [6,7,2], [9,7,9]])
y_train = np.array([1,1,1,1,1,0,0,0,0,0])
std_scale=preprocessing.StandardScaler().fit(X_train)
X_train = std_scale.transform(X_train)
print(type(y_train))
print(np.where(y_train == 1))


# In[4]:


svc = SVC(kernel='rbf',class_weight = 'balanced',degree=10 ,C = 10, gamma='scale', probability=True, cache_size = 20000)
pue = WeightedElkanotoPuClassifier(estimator=svc, labeled = 5, unlabeled = 5, hold_out_ratio = 0.2)
pue.fit(X_train,y_train)


# In[5]:


X_test = np.array([[1,0.5,1], [2,3,0.5],[3,4,2],[6,2,7],[7,8,10],[9,5,4],[8,10,3],[6,5.8,10]])
std_scale=preprocessing.StandardScaler().fit(X_test)
X_test = std_scale.transform(X_test)


# In[6]:


pue_prob_result = pue.predict_proba_experimental_version(X_test)
print(pue_prob_result)
print(np.shape(pue_prob_result))


# In[7]:


print(pue.predict_exp_version(X_test))


# In[8]:


print(pue.predict_proba_experimental_version(X_test))


# In[80]:


X_train_expe = [[1,1], [1,2],[2,4],[2,2],[3,3],[9,10],[8,5],[6,7], [7,10], [9,7]]
y_train_expe = [1,1,1,1,1,0,0,0,0,0]
std_scale=preprocessing.StandardScaler().fit(X_train_expe)
X_train_expe = std_scale.transform(X_train_expe)
data_set_expe = format_dataset(X_train_expe, y_train_expe)
X_train_expe = data_set_expe.data
y_train_expe = data_set_expe.target
X_train_expe = np.array(X_train_expe)
y_train_expe = np.array(y_train_expe)

svc = SVC(kernel='rbf',class_weight = 'balanced',degree=10 ,C = 10, gamma='scale', probability=True, cache_size = 20000)
pue_2 = WeightedElkanotoPuClassifier(estimator=svc, labeled = 5, unlabeled = 5, hold_out_ratio = 0.2)
pue_2.fit(X_train_expe,y_train_expe)
X_test_expe = np.array([[1,0.5], [1,3],[2,3.7],[2,2.2],[7,8],[9,5],[8,10],[8.6,10]])
y_test_expe = [1,1,1,1,1,0,1,0]
# X_test_expe = np.array([[1,1], [1,2],[2,4],[2,2],[3,3],[9,10],[8,5],[6,7], [7,10], [9,7]])
std_scale=preprocessing.StandardScaler().fit(X_test_expe)
X_test_expe = std_scale.transform(X_test_expe)
pue_prob_result_exp = pue_2.predict_proba_experimental_version(X_test_expe)
print(pue_prob_result_exp)
pue_predict = pue_2.predict_exp_version(X_test_expe)
print(pue_predict)


# In[81]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test_expe, pue_prob_result_exp)


# In[82]:


print("precision")
print(precision)
print("")
print("recall")
print(recall)
print("")
print("thresholds")
print(thresholds)


# In[85]:


dis_1 = plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.plot(range(0,1))


# In[84]:


pue_un = ElkanotoPuClassifier(estimator=svc, hold_out_ratio = 0.3)
pue_un.fit(X_train_expe, y_train_expe)
print(pue_un.predict(X_test_expe))
pue_un.predict_proba(X_test_expe)


# In[ ]:




