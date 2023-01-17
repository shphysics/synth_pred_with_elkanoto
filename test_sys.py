#!/usr/bin/env python
# coding: utf-8

# In[26]:


#import
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
from sklearn.model_selection import train_test_split

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


# In[ ]:


print("complete")

