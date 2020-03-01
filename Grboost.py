# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 09:03:41 2019

@author: USER
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
from sklearn.externals import joblib
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler


df = pd.read_excel('C:/Users/USER/Desktop/元大/L2R報告/Data1_2016.xlsx') #讀擋
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
    
df2 = pd.read_excel('C:/Users/USER/Desktop/元大/L2R報告/Data1_2017.xlsx') #讀擋
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

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

learning_rate=0.05
gb_clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_features=2, max_depth=2, random_state=0)
gb_clf.fit(X_train, y_train)
predictions = gb_clf.predict(X_test)
print("Learning rate: ", learning_rate)
print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))


def plot_confusion_matrix(confusion_matrix, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


cmdt = confusion_matrix(y_test, predictions)
plot_confusion_matrix(cmdt,[0,1])
#extract
df2 = pd.read_excel('C:/Users/USER/Desktop/元大/L2R報告/Data1_2017.xlsx') #原始檔案 才可抓profit
df2 = df2.replace('- -', np.nan) #替代成nan
df2 = df2.fillna(df2.mean()) #用每列平均代替

profit = []
pickrow = []

for i in range(len(predictions)):
   if predictions[i] ==0 : #test資料中 預測等於0
       #print(i)
       print('屬於0的個股:',df2.iloc[i,0])
       print('報酬率:',df2.iloc[i,2])
       profit.append(df2.iloc[i,2])
       pickrow.append(i)
       
   elif predictions[i] ==1 :#test資料中 預測等於
       #print(i)
       print('屬於1的個股:',df2.iloc[i,0])
       print('報酬率:',df2.iloc[i,2])
       profit.append(df2.iloc[i,2])

print('平均報酬率:',sum(profit)/len(profit))