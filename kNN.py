# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:27:39 2019

@author: USER
"""

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

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

num_of_neighbor = [i for i in range(1,50)]
acc = []
for i in num_of_neighbor:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn_clf=knn.fit(X_train,y_train)
    test_y_predicted = knn_clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, test_y_predicted)
    acc.append(accuracy)

plt.plot(num_of_neighbor,acc)
plt.show()
print('k for max acc:',acc.index(max(acc)),'\nMax acc:',max(acc))

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

neigh = KNeighborsClassifier(n_neighbors= acc.index(max(acc) ) )
neigh.fit(X_train,y_train)
y_pred = neigh.predict(X_test)
cmdt = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cmdt,[0,1])
print(classification_report(y_test, y_pred))

#extract
df2 = pd.read_excel('C:/Users/USER/Desktop/元大/L2R報告/Data1_2018.xlsx') #原始檔案 才可抓profit
df2 = df2.replace('- -', np.nan) #替代成nan
df2 = df2.fillna(df2.mean()) #用每列平均代替

profit = []

for i in range(len(y_pred)):
   if y_pred[i] ==0 : #test資料中 預測等於0
       #print(i)
       print('屬於0的個股:',df2.iloc[i,0])
       print('報酬率:',df2.iloc[i,2])
       profit.append(df2.iloc[i,2])
       
   elif y_pred[i] ==1 :#test資料中 預測等於
       #print(i)
       print('屬於1的個股:',df2.iloc[i,0])
       #print(y_test.iloc[i])
       print('報酬率:',df2.iloc[i,2])
       profit.append(df2.iloc[i,2])

print('平均報酬率:',sum(profit)/len(profit))
