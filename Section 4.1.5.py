#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In[] 
###Library input

import numpy as np        ###Numpy 
import pandas as pd       ###Pandas 
import matplotlib.pyplot as plt   ##Matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn import preprocessing 

from sklearn.model_selection import train_test_split   ###Split train and test data

####Machine learning libraries which is represented as scikit-learn in python
from sklearn import svm          ###Support Vector Machine
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit


from sklearn.metrics import confusion_matrix    ###To see the classified results

# In[] 
"""
Load the self-collected dataset as the input(data collected from the experiments)
"""
feature = pd.read_excel('Data collection.xlsx',sheet_name='Inorder')
labels = np.array(feature['TCV']) # split label and features
feature = feature.drop('TCV', axis = 1)
feature_list = list(feature.columns) 


# In[]
"""
Split data into training and testing sets randomly
"""
x_train, x_test, y_train, y_test = train_test_split(feature,labels, test_size=0.2)

# In[]
"""
Two loops to see how both Cost and Gamma affect the accuracy
"""
C_range = list(np.linspace(0.1,10,10))
gamma_range = list(np.linspace(0.1,1,20))

accsvm_te=np.zeros((len(C_range),len(gamma_range)))
accsvm_tr=np.zeros((len(C_range),len(gamma_range)))

##SVM Classifier 
for i in C_range:
    for j in gamma_range:
        print('i=%.2f'%i,'j=%.2f'%j)
        model = svm.SVC(C=i,kernel='rbf', gamma=j,   ###Cost and gamma can be modified
                        decision_function_shape='ovr',random_state=0) 
        model.fit(x_train, y_train)
        accsvm_te[C_range.index(i)][gamma_range.index(j)]=model.score(x_test,y_test)   
        accsvm_tr[C_range.index(i)][gamma_range.index(j)]=model.score(x_train,y_train)  


maximum=np.max(accsvm_te)
row,col=np.where(accsvm_te == maximum)
print('Maximum accuracy=%.3f'%maximum,'with cost=%d'%C_range[row[0]],'and gamma=%d'%gamma_range[col[0]])

#%matplotlib qt5
fig = plt.figure(figsize=(8,5))
ax = fig.gca(projection='3d')
x,y = np.meshgrid(gamma_range,C_range)
surf = ax.plot_surface(x,y, accsvm_te, cmap=cm.jet,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink = 0.6,aspect=20)
ax.set_xlabel('Gamma',fontsize=12, weight="bold")
ax.set_ylabel('Cost',fontsize=12, weight="bold")
ax.set_zlabel('Accuracy',fontsize=12, weight="bold")
ax.elev=23
ax.azim=-113
plt.tight_layout()
plt.show()

fig=plt.figure(figsize=(5,3))
plt.plot(C_range,np.mean(accsvm_te,axis=1),marker='o',linewidth=4.0,markersize=8)
plt.xlabel('Cost(C)',fontsize=12,weight="bold")
plt.ylabel('Accuracy',fontsize=12,weight="bold")
plt.xticks(C_range)
plt.grid()
plt.tight_layout()
#plt.savefig('Fig.15 Parameter tuning for the SVM classifier.jpg',dpi=300)
plt.show()
# In[] 
"""
Validation curve - SVM
"""
param_range = np.array(np.linspace(0.01,1,20))
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
train_scores, test_scores = validation_curve(
    SVC(), feature, labels, param_name="gamma", param_range=param_range,
    cv=cv, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fig=plt.figure(figsize=(8,3))
plt.xlabel(r"Gamma ($\Gamma$)",fontsize=14,fontweight="bold")
plt.ylabel("Score",fontsize=14,fontweight="bold")
plt.ylim(0.5, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean,'o-',color="darkorange",linewidth=3, label="Training score", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.2,color="darkorange", lw=lw)

plt.plot(param_range, test_scores_mean,'o-',color="navy",linewidth=3, label="Cross-validation score", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std,alpha=0.2,color="navy", lw=lw)
plt.xticks(np.around(param_range,decimals=2),rotation=40)
plt.legend(loc="lower right",fontsize=13)
plt.grid()
plt.tight_layout()
#plt.savefig('Fig.16(iii) Validation curves are obtained by varying gamma within a small range.jpg',dpi=300)
plt.show()


# In[] 
"""
Learning curve
"""
X = feature
y = labels
gamma = [0.001,0.1,0.53,1]

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
train_sizes=np.linspace(.1, 1.0, 7)

train_scores_mean=np.zeros((len(train_sizes),len(gamma)))
train_scores_std=np.zeros((len(train_sizes),len(gamma)))
test_scores_mean=np.zeros((len(train_sizes),len(gamma)))
test_scores_std=np.zeros((len(train_sizes),len(gamma)))


for i in gamma:
    print(i)
    estimator = SVC(gamma=i)
    train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=1, train_sizes=train_sizes)
    
    train_scores_mean[:,gamma.index(i)] = np.mean(train_scores, axis=1)
    train_scores_std[:,gamma.index(i)] = np.std(train_scores, axis=1)
    test_scores_mean[:,gamma.index(i)] = np.mean(test_scores, axis=1)
    test_scores_std[:,gamma.index(i)] = np.std(test_scores, axis=1)


fig=plt.figure(figsize=(6.5,8.5))
col=1
rows=len(gamma)
tit1=['(i.1)','(i.2)','(i.3)','(i.4)']
for i in range(len(gamma)):
    plt.subplot(rows,col,i+1)
    plt.fill_between(train_sizes, train_scores_mean[:,i] - train_scores_std[:,i],
                     train_scores_mean[:,i] + train_scores_std[:,i], alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean[:,i] - test_scores_std[:,i],
                     test_scores_mean[:,i] + test_scores_std[:,i], alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean[:,i], 'o-', color="r",linewidth=3.0,label="Training score")
    plt.plot(train_sizes, test_scores_mean[:,i], 'o-', color="g",linewidth=3.0,label="Cross-validation score")
    plt.title("%s"%tit1[i] + "Gamma=%.2f"%gamma[i],fontsize=14)
    plt.ylabel("Score",fontsize=14,fontweight="bold")
    plt.ylim(0.8, 1.01)
    plt.yticks(np.arange(0.8,1.01,step=0.05))
    plt.xticks(np.arange(2000,25000,step=2000))
    plt.grid() 
    plt.legend(ncol=2,loc="best",fontsize=10)
plt.xlabel("Training samples",fontsize=14,fontweight="bold")
plt.tight_layout()
#plt.savefig('Fig.17(i) The influence of gamma - learning curve.jpg',dpi=300)
plt.show()


# In[]
"""
A standard SVM classifier (cost & gamma can be mannually modifided)
plot confusion matrix
"""

cost=1.0  # default value
gamma = [0.001,0.1,0.53,1]
tracc=np.zeros((1,len(gamma)))
teacc=np.zeros((1,len(gamma)))

y_train_predict=np.zeros((len(y_train),len(gamma)))
y_test_predict=np.zeros((len(y_test),len(gamma)))
for i in gamma:
    print (i)
    
    model = SVC(C=1.0,kernel='rbf', gamma=i)  ###Cost and gamma can be modified
                
    # Fit the model
    model.fit(x_train, y_train)
    
    # Predicted values for training and testing sets
    y_train_predict[:,gamma.index(i)]= model.predict(x_train)
    y_test_predict[:,gamma.index(i)] = model.predict(x_test)
    
    # Mean accuracy on the testing data and labels.
    teacc[:,gamma.index(i)]=model.score(x_test,y_test) 
    # Mean gamma on the training data and labels.
    tracc[:,gamma.index(i)]=model.score(x_train,y_train)
    
# Confusion matrix
rows=4
cols=1
tit2=['(ii.1)','(ii.2)','(ii.3)','(ii.4)']
fig=plt.figure(figsize = (4,20))
for i in range(len(gamma)):
        
    C2= confusion_matrix(y_test, y_test_predict[:,i],normalize='true')
    
    df_cm = pd.DataFrame(C2, index = [i for i in ['Cold','Neutral','Hot']],
                      columns = [i for i in ['Cold','Neutral','Hot']])
    plt.subplot(rows,cols,i+1)
    ax=sns.heatmap(df_cm,cmap=cm.Blues,annot=True,annot_kws={"size":14})
    ax.set_ylim([3, 0])
    plt.title('%s. Gamma=%.2f (acc=%.2f)'%(tit2[i],gamma[i],teacc[:,i]),fontsize=12,weight="bold")
    plt.xlabel('Predicted',fontsize=11)
    plt.ylabel('Actual',fontsize=11)
plt.tight_layout()
#plt.savefig('Fig.17(ii) The influence of gamma - confusion matrices.jpg',dpi=300)
plt.show()

