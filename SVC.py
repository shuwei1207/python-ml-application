# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:27:20 2019

@author: USER
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
# ignore FutureWarning, 'cause it's so annoying 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets

df = pd.read_excel('C:/Users/USER/Desktop/元大/L2R報告/Data1_2017.xlsx') #讀擋
df = df.replace('- -', np.nan) #替代成nan
df = df.fillna(df.mean()) #用每列平均代替

    #label
for i in range(len(df)):
    if (df.iloc[i,2] >= 50):
        df = df.replace( df.iloc[i,2] , 0)
    elif (10 < df.iloc[i,2] < 50):
        df = df.replace( df.iloc[i,2] , 1)
    elif (0 <= df.iloc[i,2] <= 10):
        df = df.replace( df.iloc[i,2] , 2)
    elif (-25 < df.iloc[i,2] < 0):
        df = df.replace( df.iloc[i,2] , 3)
    else:
        df = df.replace( df.iloc[i,2] , 4)
        
col_list = list(df) #置換
df.columns = col_list
cols = list(df)
cols[2], cols[0] = cols[0], cols[2]
df = df.ix[:,cols]
    
df2 = pd.read_excel('C:/Users/USER/Desktop/元大/L2R報告/Data1_2018.xlsx') #讀擋
df2 = df2.replace('- -', np.nan) #替代成nan
df2 = df2.fillna(df2.mean()) #用每列平均代替

    #label
for i in range(len(df2)):
    if (df2.iloc[i,2] >= 50):
        df2 = df2.replace( df2.iloc[i,2] , 0)
    elif (10 < df2.iloc[i,2] < 50):
        df2 = df2.replace( df2.iloc[i,2] , 1)
    elif (0 <= df2.iloc[i,2] <= 10):
        df2 = df2.replace( df2.iloc[i,2] , 2)
    elif (-25 < df2.iloc[i,2] < 0):
        df2 = df2.replace( df2.iloc[i,2] , 3)
    else:
        df2 = df2.replace( df2.iloc[i,2] , 4)
        
col_list = list(df2) #置換
df2.columns = col_list
cols = list(df2)
cols[2], cols[0] = cols[0], cols[2]
df2 = df2.ix[:,cols]

X_train = df.iloc[:,3:53]
y_train = df.iloc[:,0]
X_test = df2.iloc[:,3:53]
y_test = df2.iloc[:,0]

#pd.DataFrame(svc_grid.cv_results_).sort_values('rank_test_score')
#best_lr_clf = lr_grid.best_estimator_

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
svc_grid = GridSearchCV(svc, parameters)
svc_grid.fit(X_train, y_train)
best_svc_clf = svc_grid.best_estimator_

print(best_svc_clf)
print('svc acc:',svc_grid.best_score_)

