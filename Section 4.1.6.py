#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np        ###Numpy 
import pandas as pd       ###Pandas 
import matplotlib.pyplot as plt   ##Matplotlib

from sklearn import preprocessing     ### It can import several kinds of data preprocessing methods

from sklearn.model_selection import train_test_split   ###Split train and test data
####Machine learning libraries which is represented as scikit-learn in python
from sklearn.model_selection import cross_val_score
from sklearn import svm                                      ###Support Vector Machine
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit

import time
from sklearn import metrics 

# In[] 
"""
Load the self-collected dataset as the input(data collected from the experiments)
"""
feature = pd.read_excel('Data collection.xlsx',sheet_name='Disorder')
#Split label and features
labels = np.array(feature['TCV'])         # labels
feature = feature.drop('TCV', axis = 1)   # input features
feature_list = list(feature.columns)      # the name of features

# In[]
"""
Split data into training and testing sets randomly
"""
x_train, x_test, y_train, y_test = train_test_split(feature,labels, test_size=0.2)

# In[]
"""
Feature importance
"""

def important_feats(x_train, y_train):
        model = RandomForestClassifier()
        model.fit(x_train,y_train)
        #plot graph of feature importances for better visualization
        plt.figure(figsize=(6,3))
        feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
        feat_importances.nlargest(x_train.shape[1]).plot(kind='barh',color='b',fontsize=12)
        plt.tight_layout()
        important_feats=feat_importances.nlargest(x_train.shape[1])
#        plt.savefig('Fig.18 Feature importance based on RF algorithm.jpg',dpi=300)
        return important_feats
    
    #identify the top i important features and form the new test and train split
important_feats = important_feats(x_train, y_train)
print(important_feats)

# In[]
"""
The effects of input features
"""
nfeature = list(range(1,5,1))
epoches = list(range(1,10))
rfte_impf=np.zeros((len(epoches),len(nfeature)))
rftr_impf=np.zeros((len(epoches),len(nfeature)))
rfcomputation_time=np.zeros((len(epoches),len(nfeature)))
for j in epoches:
    for i in nfeature:
        time_start=time.time()
        x_train, x_test, y_train, y_test = train_test_split(feature,labels, test_size=0.2)
        x_train_new=x_test_new=[]
        imporfeature_list = important_feats.index[0:i].tolist()
        print(imporfeature_list)
        x_train_new = x_train[imporfeature_list]
        x_test_new = x_test[imporfeature_list]
        # train model 
        tree_num=100 # Tree number
        tee_depth=12  # Tree depth
        rfr = RandomForestClassifier(n_estimators= tree_num, max_features = "auto", oob_score = True,max_depth=tee_depth,random_state=50) 
        rfr.fit(x_train_new, y_train)
        rfte_impf[epoches.index(j)][nfeature.index(i)]=rfr.score(x_test_new,y_test)
        rftr_impf[epoches.index(j)][nfeature.index(i)]=rfr.score(x_train_new,y_train)
        time_end=time.time()
        rfcomputation_time[epoches.index(j)][nfeature.index(i)]=time_end-time_start
        print(rfte_impf,rftr_impf)
        print('time cost',rfcomputation_time,'s')
        

svmte_impf=np.zeros((len(epoches),len(nfeature)))
svmtr_impf=np.zeros((len(epoches),len(nfeature)))
svmcomputation_time=np.zeros((len(epoches),len(nfeature)))
for j in epoches:
    for i in nfeature:
        time_start=time.time()
        x_train_new=x_test_new=[]
        x_train, x_test, y_train, y_test = train_test_split(feature,labels, test_size=0.2)
        imporfeature_list = important_feats.index[0:i].tolist()
        print(imporfeature_list)
        x_train_new = x_train[imporfeature_list]
        x_test_new = x_test[imporfeature_list]
        # train model 
        cost=1 #cost
        gamma=0.1 #gamma
        model = svm.SVC(C=cost,kernel='rbf', gamma=gamma)   ###Cost and gamma can be modified
                        
        model.fit(x_train_new, y_train)
        svmte_impf[epoches.index(j)][nfeature.index(i)]=model.score(x_test_new,y_test)
        svmtr_impf[epoches.index(j)][nfeature.index(i)]=model.score(x_train_new,y_train)
        time_end=time.time()
        svmcomputation_time[epoches.index(j)][nfeature.index(i)]=time_end-time_start
        print(svmte_impf,svmtr_impf)
        print('time cost',svmcomputation_time,'s')
        
rfte_mean = np.mean(rfte_impf,axis=0)
rftr_mean = np.mean(rftr_impf,axis=0)
rftime_mean = np.mean(rfcomputation_time,axis=0)
svmte_mean = np.mean(svmte_impf,axis=0)
svmtr_mean = np.mean(svmtr_impf,axis=0)
svmtime_mean = np.mean(svmcomputation_time,axis=0)


fig=plt.figure(figsize=(8,6)) # width, height in inches
plt.subplot(3,1,1)
plt.plot(nfeature,rfte_mean,color='royalblue',linewidth=3.5,marker='o', markerfacecolor='r',markersize=6)
plt.plot(nfeature,rftr_mean,color='royalblue',linestyle='--',linewidth=3.5,marker='v', markerfacecolor='red',markersize=8)
plt.plot(nfeature,svmte_mean,'g',linewidth=3.5,marker='o', markerfacecolor='red',markersize=8)
plt.plot(nfeature,svmtr_mean,'g--',linewidth=3.5,marker='v', markerfacecolor='red',markersize=8)
plt.ylabel('Accuracy',fontsize=14,fontweight='bold')
plt.yticks(np.arange(0.65,1.05,step=0.05))
plt.xticks(np.arange(1,5,step=1))
plt.legend(['RF-testing','RF-training','SVM-testing','SVM-training'],fontsize=11,ncol=4,loc='lower right')
plt.grid()

plt.subplot(3,1,2)
plt.plot(nfeature,rftime_mean,color='royalblue',linewidth=3.5,marker='o', markerfacecolor='r',markersize=6)
plt.ylabel('Time cost(s)',fontsize=14,fontweight='bold')
#plt.yticks(np.arange(4.5,6,step=0.5))
plt.xticks(np.arange(1,5,step=1))
plt.legend(['RF'],fontsize=13,ncol=2,loc="upper left")
plt.grid()

plt.subplot(3,1,3)
plt.plot(nfeature,svmtime_mean,color='g',linewidth=3.5,marker='v', markerfacecolor='red',markersize=8)
plt.xlabel('Number of features',fontsize=14,fontweight='bold')
plt.ylabel('Time cost(s)',fontsize=14,fontweight='bold')
#plt.yticks(np.arange(45,110,step=10))
plt.xticks(np.arange(1,5,step=1))
plt.legend(['SVM'],fontsize=13,ncol=2,loc="upper left")
plt.grid()

plt.tight_layout()
#plt.savefig('Fig.19 The influence of different number of features.jpg',dpi=300)
plt.show()

# In[]
"""
The effect of features in different combinations
"""
set_5=[['Air temperature','Skin temperature','Heart rate'], # Set No.1
      ['Air temperature','Skin temperature','GSR'],         # Set No.2
      ['Air temperature','Heart rate','GSR'],               # Set No.3
      ['Skin temperature','Heart rate','GSR'],              # Set No.4
      ['Air temperature','Skin temperature','Heart rate','GSR']] # Set No.5

rounds = list(range(1,10))
rfte_diffset=np.zeros((len(epoches),len(set_5)))
rftr_diffset=np.zeros((len(epoches),len(set_5)))
for j in rounds:
    x_train, x_test, y_train, y_test = train_test_split(feature,labels, test_size=0.2)
    for i in range(len(set_5)):
        x_train_new=x_test_new=[]
        x_train_new = x_train[set_5[i]]
        x_test_new = x_test[set_5[i]]
        # train model 
        tree_num=100 # Tree number
        tee_depth=12  # Tree depth
        rfr = RandomForestClassifier(n_estimators= tree_num, max_features = "auto", oob_score = True,max_depth=tee_depth,random_state=50) 
        rfr.fit(x_train_new, y_train)
        rfte_diffset[rounds.index(j)][i]=rfr.score(x_test_new,y_test)
        rftr_diffset[rounds.index(j)][i]=rfr.score(x_train_new,y_train)
        print(rfte_diffset,rftr_diffset)


svmte_diffset=np.zeros((len(rounds),len(set_5)))
svmtr_diffset=np.zeros((len(rounds),len(set_5)))
for j in rounds:
    x_train, x_test, y_train, y_test = train_test_split(feature,labels, test_size=0.2)
    for i in range(len(set_5)):
        x_train_new=x_test_new=[]
        x_train_new = x_train[set_5[i]]
        x_test_new = x_test[set_5[i]]
        # train model 
        cost=1 #cost
        gamma=0.1 #gamma
        model = svm.SVC(C=cost,kernel='rbf', gamma=gamma)   ###Cost and gamma can be modified
        model.fit(x_train_new, y_train)
        svmte_diffset[rounds.index(j)][i]=model.score(x_test_new,y_test)
        svmtr_diffset[rounds.index(j)][i]=model.score(x_train_new,y_train)
        print(svmte_diffset,svmtr_diffset)


rfte_diffset_mean = np.mean(rfte_diffset,axis=0)
rftr_diffset_mean = np.mean(rftr_diffset,axis=0)
svmte_diffset_mean = np.mean(svmte_diffset,axis=0)
svmtr_diffset_mean= np.mean(svmtr_diffset,axis=0)


## plot the bar graph
import matplotlib as mpl
mpl.use('Agg')
fig_size = (7,4) 

model = ('RF', 'SVM')
subjects = ('1', '2', '3','4','5')
scores = (rfte_diffset_mean, svmte_diffset_mean)

mpl.rcParams['figure.figsize'] = fig_size

bar_width = 0.2
index = np.arange(len(scores[0]))

# accuracy of RF
rects1 = plt.bar(index, scores[0], bar_width, color='#0072BC')
            
# accuracy of SVM
rects2 = plt.bar(index + bar_width, scores[1], bar_width, color='#ED1C24')
                 
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, round(height,2), ha='center', va='bottom',weight="bold")
       # set the edge color of bar to white
        rect.set_edgecolor('white')

add_labels(rects1)
add_labels(rects2)

plt.xticks(index + bar_width, subjects,fontsize=13)
plt.yticks(np.arange(0.0,1.1,step=0.1),fontsize=12)
plt.xlabel('Set No.',fontsize=14,weight="bold")
plt.ylabel('Accuracy',fontsize=14,weight="bold")
plt.legend(model,loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, ncol=5,fontsize=14)
plt.tight_layout()
#plt.savefig('Fig.20 Visualisation of the accuracy given by different sets',dpi=300)