import numpy as np
import statistics as stat
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
# In[2]:


# A bunch of functions here
#feature dictionary
elements_data = pd.read_csv('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/features_csv.csv')
elements = elements_data['Symbol']
atomic_numbers = elements_data['Number']
feature_names = elements_data.columns.values[2:]

features_dict = dict()

for item in feature_names:
    features_dict[item] = dict(zip(elements, elements_data[item]))


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

# def get_labels_pos_only(pos):
#     y = []
#     for i in range(len(pos)):
#         y.append(1)
#     return np.array(y)


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

#get data
#positive examples 
with open('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/random_in_icsd.txt', 'r') as f:
    pos = f.readlines()
    pos = [x.strip() for x in pos]
    print(len(pos))
    #10899
    
#positive examples cp 
with open('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/random_in_icsd.txt', 'r') as f:
    pos_cp = f.readlines()
    pos_cp = [x.strip() for x in pos_cp]
    print(len(pos))
    #10899
    
#unlabled data    
with open('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/random_no_icsd.txt','r') as f:
    unda = f.readlines()
    unda = [x.strip() for x in unda]
    print(len(unda))
    #2458351
    
def shuffle(lst_of_tuples, size = -1):
    random.seed(a=5)
    length = np.size(lst_of_tuples, 0)
    indices = np.arange(length)
    np.random.shuffle(indices)
    new_lst = []
    for x in indices:
        new_lst.append(lst_of_tuples[x])
    if size == -1:
        return Bunch(data=new_lst)
    return Bunch(data=new_lst[0:size])

def unpack_tuple_matname(X_samp_raw_y):
    matname_lst = []
    for i in range(len(X_samp_raw_y)):
        matname_lst.append(X_samp_raw_y[i][0])
    return matname_lst

def unpack_tuple_label(X_samp_raw_y):
    label_lst = []
    for i in range(len(X_samp_raw_y)):
        label_lst.append(X_samp_raw_y[i][1])
    return label_lst


# In[3]:


#using pd.read_csv to import diff groups of elements from the csv files

alkali = pd.read_csv('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/alkali_metals.csv')
alkaline = pd.read_csv('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/alkaline.csv')
lanthanoids = pd.read_csv('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/lanthanoid.csv')
actinoids = pd.read_csv('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/actinoid.csv')
transitional_metals = pd.read_csv('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/transition_metals.csv')
ptranstitional_metals = pd.read_csv('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/ptransition_metals.csv')
metaloids = pd.read_csv('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/metalloid.csv')
rec_nonmetals = pd.read_csv('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/nonmetals.csv')
nobel_gas = pd.read_csv('/home/users/willcai1/anaconda3/envs/my_pymatgen/2020_matsci_REU/1_week/noble_gas.csv')

#making list for each group of elements from the periodic table
alkali_lst = []
alkaline_lst = []
lanthanoids_lst = []
actinoids_lst = []
transitional_metals_lst = []
ptranstitional_metals_lst = []
metaloids_lst = []
rec_nonmetals_lst = []
nobel_gas_lst = []
for i in range(len(alkali['name'])):
    alkali_lst.append(alkali['name'][i].split(",")[2])
for i in range(len(alkaline['name'])):
    alkaline_lst.append(alkaline['name'][i].split(",")[2])
for i in range(len(lanthanoids['name'])):
    lanthanoids_lst.append(lanthanoids['name'][i].split(",")[2])
for i in range(len(actinoids['name'])):
    actinoids_lst.append(actinoids['name'][i].split(",")[2])
for i in range(len(transitional_metals['name'])):
    transitional_metals_lst.append(transitional_metals['name'][i].split(",")[2])
for i in range(len(ptranstitional_metals['name'])):
    ptranstitional_metals_lst.append(ptranstitional_metals['name'][i].split(",")[2])
for i in range(len(metaloids['name'])):
    metaloids_lst.append(metaloids['name'][i].split(",")[2])
for i in range(len(rec_nonmetals['name'])):
    rec_nonmetals_lst.append(rec_nonmetals['name'][i].split(",")[2])
for i in range(len(nobel_gas['name'])):
    nobel_gas_lst.append(nobel_gas['name'][i].split(",")[2])


# In[4]:


# helper function 1: match elem in a formula to its class and return the group name in <str>
def match_elem_group(formula):
    comp = mg.core.Composition(formula)
    species = comp.elements
    groups_key = ""
    for i in range(len(species)):
        element = str(species[i])
        if element in alkali_lst:
            groups_key += "alkali--"
        elif element in alkaline_lst:
            groups_key += "alkaline--"
        elif element in lanthanoids_lst:
            groups_key += "lanthanoid--"
        elif element in actinoids_lst:
            groups_key += "actinoid--"
        elif element in transitional_metals_lst:
            groups_key += "transition--"
        elif element in ptranstitional_metals_lst:
            groups_key += "ptransition--"
        elif element in metaloids_lst:
            groups_key += "metaloid--"
        elif element in rec_nonmetals_lst:
            groups_key += "nonmetal--"
        else: 
            groups_key += "nobel--"
    return groups_key


# In[5]:


# helper function 2: is used to set up dicts for pos and unda sets and using group tags as the keys
def make_group_tags_dict(a_list_of_data):
    the_dictionary = {}
    for i in range(len(a_list_of_data)):
        formula = a_list_of_data[i]
        group_tag = match_elem_group(formula)
        if not group_tag in the_dictionary:
            the_dictionary[group_tag] = []
            the_dictionary[group_tag].append(formula)
        else:
            the_dictionary[group_tag].append(formula)
    return the_dictionary


# In[ ]:


num_trail = 50
clf_total_mix_test_acc = []
unweight_total_mix_test_acc = []
weight_total_mix_test_acc = []
print("pos:unlabel = 1:50 for gbp")
for k in range(num_trail):
    random.seed(a=5)
    pos_dict = make_group_tags_dict(pos)
    unda_dict = make_group_tags_dict(unda)
    #process helper 3 (get common_keys): find the common keys shared between pos and uda dict
    common_keys = []
    pos_dict_keys = list(pos_dict.keys())
    unda_dict_keys = list(unda_dict.keys())
    for i in range(len(unda_dict_keys)):
        key = unda_dict_keys[i]
        if key in pos_dict_keys:
            common_keys.append(unda_dict_keys[i])

    x_raw_name_train_pos = []
    y_train_pos_label = []
    x_raw_name_test_pos = []
    y_test_pos_label = []

    x_raw_name_train_unda = []
    y_train_unda_label = []
    x_raw_name_test_unda = []
    y_test_unda_label = []

    pos_cp_dict = pos_dict
    unda_cp_dict = unda_dict

    for i in range(len(common_keys)):
        select_pos_lst = pos_cp_dict[common_keys[i]]
        select_unda_lst = unda_cp_dict[common_keys[i]]
        random.seed(a=3)
        tracking_pos = len(select_pos_lst)
        tracking_unda = int(len(select_pos_lst) // 2)

        len_of_unda_test_slt = ((len(select_pos_lst))*50)

        for k in range(len(select_pos_lst)):
            slt_num_pos = random.randrange(tracking_pos)
            slt_formula_pos = select_pos_lst.pop(slt_num_pos)
            if tracking_pos % 2 == 1:
                x_raw_name_train_pos.append(slt_formula_pos)
                y_train_pos_label.append(1)
            else: 
                x_raw_name_test_pos.append(slt_formula_pos)
                y_test_pos_label.append(1)
            tracking_pos -= 1

        for l in range(len_of_unda_test_slt):
            if len(select_unda_lst) == 0:
                print("nothing left in the list, can't do it")
            slt_num_unda = random.randrange(len(select_unda_lst))
            slt_formula_unda = select_unda_lst.pop(slt_num_unda)
            if tracking_unda != 0:
                x_raw_name_train_unda.append(slt_formula_unda)
                y_train_unda_label.append(0)
                tracking_unda -= 1
            else:
                x_raw_name_test_unda.append(slt_formula_unda)
                y_test_unda_label.append(0)

    x_raw_name_train_pos.extend(x_raw_name_train_unda)
    y_train_pos_label.extend(y_train_unda_label)
    X_popul_raw = x_raw_name_train_pos
    y = y_train_pos_label
    X_popul_raw_y = tuple(zip(X_popul_raw, y))
    x_raw_name_test_pos.extend(x_raw_name_test_unda)
    y_test_pos_label.extend(y_test_unda_label)
    X_popul_raw_y_test = tuple(zip(x_raw_name_test_pos, y_test_pos_label))
    X_popul_preprocessed_y = shuffle(X_popul_raw_y)
    X_popul_preprocessed_y_test = shuffle(X_popul_raw_y_test)

    X_popul_processed_y = X_popul_preprocessed_y.data
    X_popul_processed_y_test = X_popul_preprocessed_y_test.data

    X_samp_raw_y = X_popul_processed_y
    X_popul_raw_y_after_samp_extraction = X_popul_processed_y_test

    X_samp_matname = unpack_tuple_matname(X_samp_raw_y)
    X_samp_label = unpack_tuple_label(X_samp_raw_y) 
    X_raw_train = get_X_from_data(X_samp_matname)
    y_train_array = np.array(X_samp_label)
    std_scale=preprocessing.StandardScaler().fit(X_raw_train)
    X_train_array = std_scale.transform(X_raw_train)
    X_raw_test_mix_matname = unpack_tuple_matname(X_popul_raw_y_after_samp_extraction)
    y_test_mix_label = unpack_tuple_label(X_popul_raw_y_after_samp_extraction)
    X_raw_test_mix = get_X_from_data(X_raw_test_mix_matname)
    y_test_array = np.array(y_test_mix_label)
    std_scale=preprocessing.StandardScaler().fit(X_raw_test_mix)
    X_test_mix_array = std_scale.transform(X_raw_test_mix)

    count_label = 0
    count_unlabel = 0
    for i in range(len(X_samp_raw_y)):
        if X_samp_raw_y[i][1] == 1:
            count_label += 1
        else:
            count_unlabel += 1
    X_train = X_train_array
    X_test_array = X_test_mix_array
    X_test = X_test_mix_array
    y_train = y_train_array
    y_test = y_test_array

    clf = svm.SVC(kernel='rbf', C = 1.0, cache_size = 20000, probability=True)
    clf.fit(X_train, y_train)
    y_pred_svm_rbf = clf.predict(X_test)
    y_prd_svm_rbf_proba = clf.predict_proba(X_test)

    X_test_array = np.array(X_test)
    pos_in_X_test_array = 0
    for i in range(len(X_test_array)):
        if y_test[i] ==1:
            pos_in_X_test_array += 1
    clf_mix_test_acc = accuracy(y_pred_svm_rbf, y_test)[2] / pos_in_X_test_array
    clf_total_mix_test_acc.append(clf_mix_test_acc) 
    
    svc = SVC(kernel='rbf',class_weight = 'balanced',degree=10 ,C = 10, gamma='scale', probability=True, cache_size = 20000)
    pue_weight = WeightedElkanotoPuClassifier(estimator=svc, labeled = count_label, unlabeled = count_unlabel, hold_out_ratio = 0.4)
    pue_weight.fit(X_train_array,y_train_array)
    pue_weight_prob_result = pue_weight.predict_proba_experimental_version(X_test_array)
    pue_weight_predict = pue_weight.predict_exp_version(X_test_array)
    unweight_mix_test_accuracy = accuracy(pue_weight_predict, y_test)
    unweight_mix_test_acc = unweight_mix_test_accuracy[2] / pos_in_X_test_array
    unweight_total_mix_test_acc.append(unweight_mix_test_acc)
    
    pue_weight_prob_result_og = pue_weight.predict_proba_og(X_test_array)
    pue_weight_predict_og = pue_weight.predict_og(X_test_array)
    weight_mix_test_accuracy_og = accuracy(pue_weight_predict_og, y_test)
    weight_mix_test_acc_og = weight_mix_test_accuracy_og[2] / pos_in_X_test_array
    weight_total_mix_test_acc.append(weight_mix_test_acc_og)

      
    print("trial" + str(k) + " complete")


# In[ ]:


clf_avg_mix_test_acc = stat.mean(clf_total_mix_test_acc)
unweight_avg_mix_test_acc = stat.mean(unweight_total_mix_test_acc)
weight_avg_mix_test_acc = stat.mean(weight_total_mix_test_acc)

clf_stdev_mix_test_acc = stat.stdev(clf_total_mix_test_acc)
weight_stdev_mix_test_acc = stat.stdev(weight_total_mix_test_acc)
unweight_stdev_mix_test_acc = stat.stdev(unweight_total_mix_test_acc)

print("clf_mix_test_acc")
print(clf_avg_mix_test_acc)
print(clf_stdev_mix_test_acc)

print("unweight_mix_test_acc")
print(unweight_avg_mix_test_acc)
print(unweight_stdev_mix_test_acc)

print("weight_mix_test_acc")
print(weight_avg_mix_test_acc)
print(weight_stdev_mix_test_acc)


# In[ ]:




