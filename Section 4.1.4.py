#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# In[] 
###Library input

import numpy as np        ###Numpy 
import pandas as pd       ###Pandas 
import matplotlib.pyplot as plt   ##Matplotlib
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn import preprocessing

from sklearn.model_selection import train_test_split   ###Split train and test data

####Machine learning libraries which is represented as scikit-learn in python
from sklearn.model_selection import cross_val_score          
from sklearn.ensemble import RandomForestClassifier ###Random Forest
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import confusion_matrix                 ###To see the classified results


import time
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
Section 4.1.3
Obeservations about dataset
"""
# Show the visualization effect of the relationship between each feature(index) and Thermal sensation vote
nums=len(feature_list)
columns=2
rows=2
tit1=['(i)','(ii)','(iii)','(iv)']

fig=plt.figure(figsize=(10,5))
for i in range(nums):
    plt.subplot(rows,columns,i+1)
    plt.hist(feature.values[:,i], bins=14, normed=0, edgecolor="black")
    plt.title('%s'%tit1[i] +'%s'%feature_list[i], fontsize=14, weight="bold")
    plt.grid(linestyle='--')
plt.rc('axes', axisbelow=True)
plt.tight_layout()

#plt.savefig('Fig.9 Data distribution of each feature.jpg',dpi=300)

fig=plt.figure(figsize=(10,5))
for i in range(nums):
    plt.subplot(rows,columns,i+1)
    plt.scatter(feature.values[:,i],labels,marker='o',s=8)
    plt.title('%s'%tit1[i] +'%s'%feature_list[i], fontsize=14, weight="bold")
    plt.yticks(np.arange(-1,1.1,step=1))
    plt.ylabel('TCV',fontsize=12)
    plt.subplots_adjust(hspace=1.0)
    plt.grid(linestyle='--',axis='y')
plt.rc('axes', axisbelow=True)
plt.tight_layout()
#plt.savefig('Fig.10 Visualization of the relationship between each feature and TSV.jpg',dpi=300)
plt.show()

# In[]
"""
Split data into training and testing sets randomly
"""
x_train, x_test, y_train, y_test = train_test_split(feature,labels, test_size=0.2)

# In[]

"""
Using two loops to see how both tree_number and tree_depth affect the accuracy
"""
tree_number=list(range(10,200,10))
tree_depth=list(range(5,20,2))

accrf_te=np.zeros((len(tree_number),len(tree_depth)))
accrf_tr=np.zeros((len(tree_number),len(tree_depth)))

##RF Classifier 
for j in tree_number:
    for i in tree_depth:
        print('tree_depth=%d'%i,'tree_num=%d'%j)
        rfr = RandomForestClassifier(n_estimators= j, max_features = "auto", oob_score = True,max_depth=i
                                  ,random_state=0) ###Tree depth and tree numbers can be modified
        rfr.fit(x_train, y_train)
        accrf_te[tree_number.index(j)][tree_depth.index(i)]=rfr.score(x_test,y_test)
        accrf_tr[tree_number.index(j)][tree_depth.index(i)]=rfr.score(x_train,y_train)
        
maximum=np.max(accrf_te)
row,col=np.where(accrf_te==maximum)
max_treenum=tree_number[row[0]]
max_treedepth=tree_depth[col[0]]
print('Maximum accuracy=%.2f'%maximum,'with tree_number=%d'%max_treenum,'and tree_depth=%d'%max_treedepth)

#%matplotlib qt5
fig = plt.figure(figsize=(6,4))
ax = fig.gca(projection='3d')
x,y = np.meshgrid(tree_depth, tree_number)
surf = ax.plot_surface(x,y, accrf_te, cmap=cm.jet,linewidth=0, antialiased=False)
ax.set_xlabel('Tree depth',fontsize=12, weight="bold")
ax.set_ylabel('Tree number',fontsize=12, weight="bold")
ax.set_zlabel('Accuracy',fontsize=12, weight="bold",labelpad=3)
ax.elev=20
ax.azim=-146
fig.colorbar(surf, shrink = 0.6,aspect=20)
plt.tight_layout()
#plt.savefig('Fig.13(i) Parameter tuning for the RF classifier.jpg',dpi=300)
plt.show()

# In[] 
"""
Validation curve vs. tree depth
"""
param_range = np.array(range(1,21))

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
rfc = RandomForestClassifier(n_estimators=100)
train_scores, test_scores = validation_curve(
   rfc, feature, labels, param_name="max_depth", param_range=param_range,
    cv=cv, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fig=plt.figure(figsize=(8,3))
plt.title("Validation Curve with RF")
plt.xlabel("Tree depth",fontsize=14,fontweight="bold")
plt.ylabel("Score",fontsize=14,fontweight="bold")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",linewidth =3.0,marker='o',
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",linewidth =3.0,marker='o',
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.xticks(np.around(param_range,decimals=2),rotation=40)
plt.legend(loc="best")
plt.grid()
#plt.savefig('Fig.13(ii)Validation Curves - RF,tree_number=100.jpg',dpi=300)
plt.show()

# In[] 
"""
Learning curve 
"""
X = feature
y = labels
tree_depth = [1,5,8,12]
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
train_sizes=np.linspace(.1, 1.0, 7)

train_scores_mean=np.zeros((len(train_sizes),len(tree_depth)))
train_scores_std=np.zeros((len(train_sizes),len(tree_depth)))
test_scores_mean=np.zeros((len(train_sizes),len(tree_depth)))
test_scores_std=np.zeros((len(train_sizes),len(tree_depth)))


for i in tree_depth:
    print(i)
    estimator = RandomForestClassifier(max_depth=i,random_state=0)
    train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=1, train_sizes=train_sizes)
    
    
    train_scores_mean[:,tree_depth.index(i)] = np.mean(train_scores, axis=1)
    train_scores_std[:,tree_depth.index(i)] = np.std(train_scores, axis=1)
    test_scores_mean[:,tree_depth.index(i)] = np.mean(test_scores, axis=1)
    test_scores_std[:,tree_depth.index(i)] = np.std(test_scores, axis=1)


fig=plt.figure(figsize=(6.5,8.5))
col=1
rows=len(tree_depth)
tit1=['(i.1)','(i.2)','(i.3)','(i.4)']
for i in range(len(tree_depth)):
    plt.subplot(rows,col,i+1)
    plt.fill_between(train_sizes, train_scores_mean[:,i] - train_scores_std[:,i],
                     train_scores_mean[:,i] + train_scores_std[:,i], alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean[:,i] - test_scores_std[:,i],
                     test_scores_mean[:,i] + test_scores_std[:,i], alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean[:,i], 'o-', color="r",linewidth=3.0,label="Training score")
    plt.plot(train_sizes, test_scores_mean[:,i], 'o-', color="g",linewidth=3.0,label="Cross-validation score")
    plt.title("%s"%tit1[i] + "Tree depth=%d"%tree_depth[i],fontsize=14)
    plt.ylabel("Score",fontsize=14,fontweight="bold")
    plt.ylim(0.8, 1.01)
    plt.yticks(np.arange(0.8,1.01,step=0.05))
    plt.xticks(np.arange(2000,25000,step=2000))
    plt.grid() 
    plt.legend(ncol=2,loc="best",fontsize=10)
plt.xlabel("Training samples",fontsize=14,fontweight="bold")
plt.tight_layout()
#plt.savefig('Fig.14(i) Learning Curves.jpg',dpi=300)
plt.show()

# In[]
"""
A standard RF classifier (tree number & tree depth can be mannually modifided)\
plot confusion matrices
"""

tree_number=100  # Tree number
tree_depth=[1,5,8,12]    # Tree depth
tracc=np.zeros((1,len(tree_depth)))
teacc=np.zeros((1,len(tree_depth)))


y_train_predict=np.zeros((len(y_train),len(tree_depth)))
y_test_predict=np.zeros((len(y_test),len(tree_depth)))
for i in tree_depth:
    print (i)
    # Build the model
    """
     Parameters description:
     - "n_estimators": the number of trees in the forest
     - "max_features": the number of features to consider when looking for the best split; 
                       If “auto”, then max_features=sqrt(n_features).
     - "max_depth": the maximum depth of the tree 
     - "oob_scorebool", default=False: whether to use out-of-bag samples to estimate the generalization accuracy
     
     Tree number and tree depth are two main parameters to be tuned
    """
    rfr = RandomForestClassifier(n_estimators= tree_number, max_features = "auto", oob_score = True,max_depth=i) 
    
    # Fit the model
    rfr.fit(x_train, y_train)
    
    # Predicted values for training and testing sets
    y_train_predict[:,tree_depth.index(i)]= rfr.predict(x_train)
    y_test_predict[:,tree_depth.index(i)] = rfr.predict(x_test)
    
    # Mean accuracy on the testing data and labels.
    teacc[:,tree_depth.index(i)]=rfr.score(x_test,y_test) 
    # Mean accuracy on the training data and labels.
    tracc[:,tree_depth.index(i)]=rfr.score(x_train,y_train)


rows=2
cols=2
tit2=['(ii.1)','(ii.2)','(ii.3)','(ii.4)']
fig=plt.figure(figsize = (8,7))
for i in range(len(tree_depth)):
        
    C2= confusion_matrix(y_test, y_test_predict[:,i],normalize='true')
    
    df_cm = pd.DataFrame(C2, index = [i for i in ['Cold','Neutral','Hot']],
                      columns = [i for i in ['Cold','Neutral','Hot']])
    plt.subplot(rows,cols,i+1)
    ax=sns.heatmap(df_cm,cmap=cm.Blues,annot=True,annot_kws={"size": 14})
    ax.set_ylim([3, 0])
    plt.title('%s. Tree depth=%d (acc=%.2f)'%(tit2[i],tree_depth[i],teacc[:,i]),fontsize=12,weight="bold")
    plt.xlabel('Predicted',fontsize=11)
    plt.ylabel('Actual',fontsize=11)
plt.tight_layout()
#plt.savefig('Fig.14(ii) Confusion matrices.jpg',dpi=300)
plt.show()



