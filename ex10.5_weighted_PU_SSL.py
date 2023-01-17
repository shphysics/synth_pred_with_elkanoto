#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import a bunch of packages

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

from pulearn import ElkanotoPuClassifier
from pulearn import WeightedElkanotoPuClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
style.use("ggplot")

from numpy import asarray
from numpy import savetxt
from numpy import loadtxt


# In[2]:


#feature dictionary
elements_data = pd.read_csv('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/features_csv.csv')
elements = elements_data['Symbol']
atomic_numbers = elements_data['Number']
feature_names = elements_data.columns.values[2:]

print('features: {}'.format(feature_names))

features_dict = dict()

for item in feature_names:
    features_dict[item] = dict(zip(elements, elements_data[item]))


# In[3]:


#code for making features
def get_fractions(formulas):
	N = len(formulas)
	fractions = np.zeros((N, 2)) #??? what is you have more than 2 elements in a compound
	for i, formula in enumerate(formulas):
		comp = mg.core.Composition(formula)
		species = comp.elements
		for j in range(2):
			fractions[i, j] = comp.get_atomic_fraction(species[j])
	return fractions





def get_mean_features(formulas, feature):
	# print feature
	N = len(formulas)
	feature_values = np.zeros(N)
	for i, formula in enumerate(formulas):
		comp = mg.core.Composition(formula)
		k = len(mg.core.Composition(formula))
		species = list(map(str,comp.elements)) #turn the double quote into single quote
		for j in range(k):
			feature_values[i] += features_dict[feature][species[j]] * comp.get_atomic_fraction(species[j])
	return feature_values


def get_product_features(formulas, feature):
	N = len(formulas)
	feature_values = np.ones(N)  # need ones, not zeros
	for i, formula in enumerate(formulas):
		comp = mg.core.Composition(formula)
		k = len(mg.core.Composition(formula))
		species = list(map(str, comp.elements))
#		species = comp.elements (can't go this code)
		for j in range(k):
			feature_values[i] *= features_dict[feature][species[j]] * comp.get_atomic_fraction(
				species[j])
	return feature_values


def get_X_from_data(raw_formulas, feature_list=feature_names):
	formulas=raw_formulas
	# formulas=[]
	# for formula in raw_formulas:
	# 	inc=True
	# 	for e in exclude:
	# 		if e in formula:
	# 			inc=False
	# 	if inc:
	# 		formulas.append(formula)
	X = get_fractions(formulas)
	# X=np.random.randint(2, size=len(data))
	for item in feature_list:
		featureValues_mean = get_mean_features(formulas, item)
		featureValues_prod = get_product_features(formulas, item)
		# featureValues_diff = get_diff_features(data, item)
		if len(np.shape(X)) == 1:
			# X=featureValues
			np.column_stack((featureValues_mean, featureValues_prod))
		else:
			X = np.column_stack((X, featureValues_mean, featureValues_prod))
	return X

#shuffles dataset
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

#pos, neg are lists
def get_labels(pos, neg):
    y = []
    for i in range(len(pos)):
        y.append(1)
    for i in range(len(neg)):
        y.append(0)
    return np.array(y)


#returns information on success of model
def accuracy(pred, test):
    counter = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    temp = 0
    for i in range(0,np.size(pred)):
        if pred[i] < 0.5:
            temp = 0
        else:
            temp = 1
        if test[i] == temp:
            counter += 1
        if test[i] == 1 and temp == 1:
            true_positive += 1
        if test[i] == 0 and temp == 1:
            false_positive += 1
        if test[i] == 1 and temp == 0:
            false_negative += 1
        if test[i] == 0 and temp == 0:
            true_negative += 1
    return [counter, np.size(pred), true_positive, false_positive, false_negative, true_negative]


# In[4]:


#get data
with open('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/random_in_icsd.txt', 'r') as f:
    pos = f.readlines()
    pos = [x.strip() for x in pos]
#     print(len(pos))
    #10899
with open('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/random_no_icsd.txt', 'r') as f:
    neg = f.readlines()
    neg = [x.strip() for x in neg]
#     print(len(neg))
    #2458351


# In[5]:


#featurize the positive examples
#make features from formulas_list
random.seed(a=5)
# num_examples=100

#computationally intensive
num_examples = 544950

# do some samplings
# pos_sample=random.sample(pos, num_examples)

#use the whole positive population
pos_sample = pos
X_pos = get_X_from_data(pos_sample)

# print(np.shape(X_pos))


# In[6]:


#featurize negative examples, set here to be the same length as the number of positive examples
random.seed(a=5)
neg_sample=random.sample(neg, num_examples)
X_neg=get_X_from_data(neg_sample)

#concatenate the features together and scale
X = np.concatenate((X_pos, X_neg), axis = 0)
std_scale=preprocessing.StandardScaler().fit(X)
X = std_scale.transform(X)

#get output label
y = []

#Test 1: positive = ICSD, negative = random
#Test 2: positive = random same format, negative = random

y=get_labels(pos_sample,neg_sample)

# print(np.shape(X))


# In[7]:


#90% for traning and 15% for testing
train = 500264
syn = format_dataset(X, y)

#Split the data into training/testing sets.
portion = train
X_train = syn.data[:portion]

X_test = syn.data[portion:]

# Split the targets into training/testing sets
y_train = syn.target[:portion]
y_test = syn.target[portion:]
#print(y_train)

# Create regression object
#lin_reg = linear_model.LinearRegression()
#log_reg = linear_model.LogisticRegression()
#svm_lin = svm.LinearSVC()
#las_reg = linear_model.Lasso(alpha=0.1)
#rid_reg = Ridge(alpha=1.0)
#nai_bay = GaussianNB()
#ran_for = RandomForestClassifier(n_estimators=100)
#dec_tre = tree.DecisionTreeClassifier()
clf = svm.SVC(kernel='rbf', C = 1.0, cache_size = 20000, probability=True)

# Train the model using the training sets
#lin_reg.fit(X_train, y_train)
#log_reg.fit(X_train, y_train)
#svm_lin.fit(X_train, y_train)
#las_reg.fit(X_train, y_train)
#rid_reg.fit(X_train, y_train)
#nai_bay.fit(X_train, y_train)
#ran_for.fit(X_train, y_train)
#dec_tre.fit(X_train, y_train)

# Make predictions using the testing set
#y_pred_lin = lin_reg.predict(X_test)


#calculate linear regression accuracy
#curr_accuracy = accuracy(y_pred_svm_li, y_test)[0] / accuracy(y_pred_svm_li, y_test)[1]
#print(curr_accuracy)

# print(len(X_train))
# print(np.shape(X_test))

# print(np.shape(y_train))
# print(np.shape(y_test))


# In[8]:


clf.fit(X_train, y_train)


# In[9]:


y_pred_svm_rbf = clf.predict(X_test)
y_prd_svm_rbf_proba = clf.predict_proba(X_test)


# In[12]:


X_test_array = np.array(X_test)
pos_in_X_test_array = 0
for i in range(len(X_test_array)):
    if y_test[i] ==1:
        pos_in_X_test_array += 1

curr_accuracy = accuracy(y_pred_svm_rbf, y_test)[0] / accuracy(y_pred_svm_rbf, y_test)[1]
print("normal_rbf_based_SVM_accuracy_under_normal_assumptions")
print(curr_accuracy)
clf_total_mix_test_acc = accuracy(y_pred_svm_rbf, y_test)[2] / pos_in_X_test_array
print("clf_total_mix_test_acc")
print(clf_total_mix_test_acc)

clf_pred_un_as_pos_num = accuracy(y_pred_svm_rbf, y_test)[3]
print("clf_pred_un_as_pos_num")
print(clf_pred_un_as_pos_num)
clf_pre_un_as_pos_percent = clf_pred_un_as_pos_num / len(y_test)
print("clf_pre_un_as_pos_percent")
print(clf_pre_un_as_pos_percent)


# In[13]:


precision_clf, recall_clf, thresholds_clf = precision_recall_curve(y_test, y_prd_svm_rbf_proba[:,1])


# In[14]:


count_label = 0
count_unlabel = 0
for i in range(len(y_train)):
    if y_train[i] == 1:
        count_label += 1
count_unlabel = len(y_train) - count_label
# print(count_label)
# print(count_unlabel)
# print(len(y_train))


# In[15]:


X_train_array = np.array(X_train)
y_train_array = np.array(y_train)


# In[16]:


svc = SVC(kernel='rbf',class_weight = 'balanced',degree=10 ,C = 10, gamma='scale', probability=True, cache_size = 20000)
pue_weight = WeightedElkanotoPuClassifier(estimator=svc, labeled = count_label, unlabeled = count_unlabel, hold_out_ratio = 0.4)


# In[17]:


pue_weight.fit(X_train_array,y_train_array)


# In[18]:


pue_weight_prob_result = pue_weight.predict_proba_experimental_version(X_test_array)
pue_weight_predict = pue_weight.predict_exp_version(X_test_array)


# In[19]:


all_num_pos_predict = 0
for i in range(len(pue_weight_predict)):
    if pue_weight_predict[i] == 1:
        all_num_pos_predict += 1


# In[21]:


unweight_mix_test_accuracy = accuracy(pue_weight_predict, y_test)
unweight_total_mix_test_acc = unweight_mix_test_accuracy[2] / pos_in_X_test_array
print("unweight_total_mix_test_acc")
print(unweight_total_mix_test_acc)


# In[22]:


precision_un, recall_un, thresholds_un = precision_recall_curve(y_test, pue_weight_prob_result)


# In[23]:


pred_un_to_pos_num = unweight_mix_test_accuracy[3]
print("pred_un_to_pos_num")
print(pred_un_to_pos_num)
pre_un_to_pos_percent = pred_un_to_pos_num / len(y_test)
print("pre_un_to_pos_percent")
print(pre_un_to_pos_percent)


# In[24]:


pue_weight_prob_result_og = pue_weight.predict_proba_og(X_test_array)
pue_weight_predict_og = pue_weight.predict_og(X_test_array)


# In[25]:


all_num_pos_predict_og = 0
for i in range(len(pue_weight_predict_og)):
    if pue_weight_predict_og[i] == 1:
        all_num_pos_predict_og += 1


# In[26]:


weight_mix_test_accuracy_og = accuracy(pue_weight_predict_og, y_test)
weight_total_mix_test_acc_og = weight_mix_test_accuracy_og[2] / pos_in_X_test_array
print("weight_total_mix_test_acc")
print(weight_total_mix_test_acc_og)


# In[27]:


precision_weight, recall_weight, thresholds_weight = precision_recall_curve(y_test, pue_weight_prob_result_og)


# In[28]:


pred_un_to_pos_num_og = weight_mix_test_accuracy_og[3]
print(pred_un_to_pos_num_og)
pre_un_to_pos_percent_og = pred_un_to_pos_num_og / len(y_test)
print(pre_un_to_pos_percent_og)


# In[29]:


with open('z_1:50_90%train_recall_weight.csv', 'w', newline='') as file:
    savetxt('z_1:50_90%train_recall_weight.csv', recall_weight, delimiter=',')
with open('z_1:50_90%train_precision_weight.csv', 'w', newline='') as file:
    savetxt('z_1:50_90%train_precision_weight.csv', precision_weight, delimiter=',')


# In[30]:


with open('z_1:50_90%train_recall_un.csv', 'w', newline='') as file:
    savetxt('z_1:50_90%train_recall_un.csv', recall_un, delimiter=',')
with open('z_1:50_90%train_precision_un.csv', 'w', newline='') as file:
    savetxt('z_1:50_90%train_precision_un.csv', precision_un, delimiter=',')


# In[31]:


with open('z_1:50_90%train_recall_clf.csv', 'w', newline='') as file:
    savetxt('z_1:50_90%train_recall_clf.csv', recall_clf, delimiter=',')
with open('z_1:50_90%train_precision_clf.csv', 'w', newline='') as file:
    savetxt('z_1:50_90%train_precision_clf.csv', precision_clf, delimiter=',')


# In[ ]:




