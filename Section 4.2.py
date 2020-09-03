#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In[] 
###Library input

import numpy as np        ###Numpy 
import pandas as pd       ###Pandas 
import matplotlib.pyplot as plt   ##Matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn import preprocessing     ### It can import several kinds of data preprocessing methods
from sklearn.preprocessing import StandardScaler     ### Three data preprocessing methods
from sklearn.preprocessing import Normalizer         
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split   ###Split train and test data

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit

####Machine learning libraries which is represented as scikit-learn in python
from sklearn import svm                                      ###Support Vector Machine
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix                 ###To see the classified results

import time

# In[] 
###Data input and type transformation

# Load the dataset based on three categories of thermal sensation vote

features_3point = pd.read_excel('Three category (New).xlsx')   # dataset in 3-point scale at TSV
features_7point = pd.read_excel('Seven category (New).xlsx')   # dataset in 7-point scale at TSV

labels_3point = np.array(features_3point['TS'])    # split label and features
feature_3point = features_3point.drop('TS', axis = 1)

labels_7point = np.array(features_7point['TS'])    # split label and features
feature_7point = features_7point.drop('TS', axis = 1)

featurelist_3point = list(feature_3point.columns) 
featurelist_7point = list(feature_7point.columns) 



# In[]
"""
 Compare 3-point and 7-point TSV
 Use label encoding
"""
# label encoding
features_t = ['Season','Building','Mode','Sex']
for i in features_t:
    le = preprocessing.LabelEncoder()
    le = le.fit(feature_3point[i])
    feature_3point[i] = le.transform(feature_3point[i])
    le = le.fit(feature_7point[i])
    feature_7point[i] = le.transform(feature_7point[i])

rounds=list(range(1,10,1))

rfaccte_3point=np.zeros((1,len(rounds)))
rfacctr_3point=np.zeros((1,len(rounds)))
rfaccte_7point=np.zeros((1,len(rounds)))
rfacctr_7point=np.zeros((1,len(rounds)))

svmaccte_3point=np.zeros((1,len(rounds)))
svmacctr_3point=np.zeros((1,len(rounds)))
svmaccte_7point=np.zeros((1,len(rounds)))
svmacctr_7point=np.zeros((1,len(rounds)))

# run 10 rounds and take average
for i in rounds:
        print('i=%d'%i)
        #RF
        tree_number=300
        tree_depth=7
        x_train3p, x_test3p, y_train3p, y_test3p = train_test_split(feature_3point,labels_3point, test_size=0.2)
        x_train7p, x_test7p, y_train7p, y_test7p = train_test_split(feature_7point,labels_7point, test_size=0.2)
        rfr = RandomForestClassifier(n_estimators= tree_number, max_features = "auto", oob_score = True,max_depth=tree_depth
                                  ,random_state=50) ###Tree depth and tree numbers can be modified
        # 3-point TSV 
        rfr.fit(x_train3p, y_train3p)
        rfaccte_3point[:,rounds.index(i)]=rfr.score(x_test3p,y_test3p)
        rfacctr_3point[:,rounds.index(i)]=rfr.score(x_train3p,y_train3p)

        # 7-point TSV 
        rfr.fit(x_train7p, y_train7p)
        rfaccte_7point[:,rounds.index(i)]=rfr.score(x_test7p,y_test7p)
        rfacctr_7point[:,rounds.index(i)]=rfr.score(x_train7p,y_train7p)
        
        
        #SVM
        cost=1.0
        gamma=0.1
        model = svm.SVC(C=cost,kernel='rbf', gamma=gamma,   ###Cost and gamma can be modified
                        decision_function_shape='ovr',random_state=0) 
        # 3-point TSV 
        model.fit(x_train3p, y_train3p)
        svmaccte_3point[:,rounds.index(i)]=model.score(x_test3p,y_test3p)
        svmacctr_3point[:,rounds.index(i)]=model.score(x_train3p,y_train3p)

        # 7-point TSV 
        model.fit(x_train7p, y_train7p)
        svmaccte_7point[:,rounds.index(i)]=model.score(x_test7p,y_test7p)
        svmacctr_7point[:,rounds.index(i)]=model.score(x_train7p,y_train7p)


rfaccte_3point_mean=np.mean(rfaccte_3point)
rfacctr_3point_mean=np.mean(rfacctr_3point)
rfaccte_7point_mean=np.mean(rfaccte_7point)
rfacctr_7point_mean=np.mean(rfacctr_7point)
svmaccte_3point_mean=np.mean(svmaccte_3point)
svmacctr_3point_mean=np.mean(svmacctr_3point)
svmaccte_7point_mean=np.mean(svmaccte_7point)
svmacctr_7point_mean=np.mean(svmacctr_7point)
## plot the bar graph

# In[]
"""
Compare one-hot encoding and label encoding
Use 3-point TSV
"""
## One-hot Encoding
features_t = ['Season','Building','Mode','Sex']
a=b=0
for i in features_t:
    values=feature_3point[i]
    a=len(np.unique(values))
    b=b+a
    
onehot_encoded=[]
label_encoded=[]
for i in features_t:
    values=feature_3point[i]
    label_encoder = preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    
    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded.append(onehot_encoder.fit_transform(integer_encoded))

oh_encoded=np.zeros((len(feature_3point),b+(feature_3point.shape[1]-len(features_t))))     
oh_encoded[:,0:4]=np.array(onehot_encoded[0])
oh_encoded[:,4:7]=np.array(onehot_encoded[1])
oh_encoded[:,7:10]=np.array(onehot_encoded[2])
oh_encoded[:,10:12]=np.array(onehot_encoded[3])
f=['Age','SET','Clo','Met','AirT(°C)','RH','Airspeed(m/s)','OutT (°C)']
oh_encoded[:,12:20]=feature_3point[f]


## Label Encoding
features_t = ['Season','Building','Mode','Sex']
for i in features_t:
    le = preprocessing.LabelEncoder()
    le = le.fit(feature_3point[i])
    feature_3point[i] = le.transform(feature_3point[i])

    
x_train_oh, x_test_oh, y_train_oh, y_test_oh = train_test_split(oh_encoded,labels_3point, test_size=0.2)
x_train_lb, x_test_lb, y_train_lb, y_test_lb = train_test_split(feature_3point,labels_3point, test_size=0.2)

# RF
tree_number=200  # Tree number
tree_depth=7  # Tree depth
rfr = RandomForestClassifier(n_estimators= tree_number, max_features = "auto", oob_score = True,max_depth=tree_depth,random_state=50) ###Tree depth and tree numbers can be modified
# train model respect to one-hot encoding
rfaccte_oh=0
rfacctr_oh=0
rfr.fit(x_train_oh, y_train_oh)
rfaccte_oh=rfr.score(x_test_oh,y_test_oh)
rfacctr_oh=rfr.score(x_train_oh,y_train_oh)

# train model respect to label encoding
rfaccte_lb=0
rfacctr_lb=0
rfr.fit(x_train_lb, y_train_lb)
rfaccte_lb=rfr.score(x_test_lb,y_test_lb)
rfacctr_lb=rfr.score(x_train_lb,y_train_lb)


#SVM
cost=1.0 #cost
gamma=0.1 #gamma
model = svm.SVC(C=cost,kernel='rbf', gamma=gamma,   ###Cost and gamma can be modified
                        decision_function_shape='ovr',random_state=0) 
# train model respect to one-hot encoding        
model.fit(x_train_oh, y_train_oh)
svmaccte_oh=0
svmacctr_oh=0
svmaccte_oh=model.score(x_test_oh,y_test_oh)        
svmacctr_oh=model.score(x_train_oh,y_train_oh)  

# train model respect to label encoding
svmaccte_lb=0
svmacctr_lb=0
model.fit(x_train_lb, y_train_lb)
svmaccte_lb=model.score(x_test_lb,y_test_lb)
svmacctr_lb =model.score(x_train_lb,y_train_lb)


# In[]
"""
   Compare continuous feature processing methods
   Use label encoding and 3-point TSV
"""
features_t = ['Season','Building','Mode','Sex']
for i in features_t:
    le = preprocessing.LabelEncoder()
    le = le.fit(feature_3point[i])
    feature_3point[i] = le.transform(feature_3point[i])

x_train, x_test, y_train, y_test = train_test_split(feature_3point,labels_3point, test_size=0.2)

nor_x = Normalizer()                         
x_train_nor =nor_x.fit_transform(x_train)
x_test_nor = nor_x.fit_transform(x_test)

ss_x = StandardScaler()
x_train_ss = ss_x.fit_transform(x_train)
x_test_ss  = ss_x.fit_transform(x_test)

mm_x = MinMaxScaler()
x_train_mm = mm_x.fit_transform(x_train)
x_test_mm  = mm_x.fit_transform(x_test)

tree_number=200  # Tree number
tree_depth=8  # Tree depth
rfr = RandomForestClassifier(n_estimators= tree_number, max_features = "auto", oob_score = True,max_depth=tree_depth,random_state=50) ###Tree depth and tree numbers can be modified

cost=1.0 #cost
gamma=0.1 #gamma
model = svm.SVC(C=cost,kernel='rbf', gamma=gamma,   ###Cost and gamma can be modified
                        decision_function_shape='ovr',random_state=0) 
        

# Raw data - RF
rfaccte_rd=0
rfacctr_rd=0
rfr.fit(x_train, y_train)
rfaccte_rd=rfr.score(x_test,y_test)
rfacctr_rd=rfr.score(x_train,y_train)

#  Raw data - SVM
svmaccte_rd=0
svmacctr_rd=0
model.fit(x_train, y_train)
svmaccte_rd=model.score(x_test,y_test)
svmacctr_rd=model.score(x_train,y_train)


# Normalizer() - RF
rfaccte_nor=0
rfacctr_nor=0
rfr.fit(x_train_nor, y_train)
rfaccte_nor=rfr.score(x_test_nor,y_test)
rfacctr_nor=rfr.score(x_train_nor,y_train)

# Normalizer() - SVM
svmaccte_nor=0
svmacctr_nor=0
model.fit(x_train_nor, y_train)
svmaccte_nor=model.score(x_test_nor,y_test)
svmacctr_nor=model.score(x_train_nor,y_train)



# StandardScaler() - RF
rfaccte_ss=0
rfacctr_ss=0
rfr.fit(x_train_ss, y_train)
rfaccte_ss=rfr.score(x_test_ss,y_test)
rfacctr_ss=rfr.score(x_train_ss,y_train)

# StandardScaler() - SVM
svmaccte_ss=0
svmacctr_ss=0
model.fit(x_train_ss, y_train)
svmaccte_ss=model.score(x_test_ss,y_test)
svmacctr_ss=model.score(x_train_ss,y_train)


# MinMaxScaler() - RF
rfaccte_mm=0
rfacctr_mm=0
rfr.fit(x_train_mm, y_train)
rfaccte_mm=rfr.score(x_test_mm,y_test)
rfacctr_mm=rfr.score(x_train_mm,y_train)

# MinMaxScaler() - SVM
svmaccte_mm=0
svmacctr_mm=0
model.fit(x_train_mm, y_train)
svmaccte_mm=model.score(x_test_mm,y_test)
svmacctr_mm=model.score(x_train_mm,y_train)

rfaccte_scaleunits=[rfaccte_rd,rfaccte_nor,rfaccte_ss,rfaccte_mm]
rfacctr_scaleunits=[rfacctr_rd,rfacctr_nor,rfacctr_ss,rfacctr_mm]
svmaccte_scaleunits=[svmaccte_rd,svmaccte_nor,svmaccte_ss,svmaccte_mm]
svmacctr_scaleunits=[svmacctr_rd,svmacctr_nor,svmacctr_ss,svmacctr_mm]



# In[]
plt.figure(figsize = (7,5))

plt.subplot(2,2,1)
model = ('RF', 'SVM')
subjects = ('3-point TSV','7-point TSV')
scores = ((rfaccte_3point_mean,rfaccte_7point_mean),
          (svmaccte_3point_mean,svmaccte_7point_mean))

bar_width = 0.2
index = np.arange(len(scores[0]))

# accuracy of RF
rects1 = plt.bar(index, scores[0], bar_width)
            
# accuracy of SVM
rects2 = plt.bar(index + bar_width, scores[1], bar_width)
                 
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, round(height,2), ha='center', va='bottom',weight="bold")
       # set the edge color of bar to white
        rect.set_edgecolor('white')
plt.grid(linestyle='--',axis='y')
add_labels(rects1)
add_labels(rects2)

plt.xticks(index + bar_width/2,subjects,fontsize=11)
plt.yticks(np.arange(0.0,1.1,step=0.1),fontsize=9)
plt.xlabel('Different TSV scales',fontsize=11,weight="bold")
plt.ylabel('Accuracy',fontsize=11,weight="bold")
plt.rc('axes', axisbelow=True)
plt.legend(model,loc='upper center', bbox_to_anchor=(0.5, 1.25), fancybox=True, ncol=5,fontsize=11)


plt.subplot(2,2,2)
model = ('RF', 'SVM')
subjects = ('One-hot encoding','Label encoding')
scores = ((rfaccte_oh,rfaccte_lb),
          (svmaccte_oh,svmaccte_lb))


bar_width = 0.2
index = np.arange(len(scores[0]))

# accuracy of RF
rects1 = plt.bar(index, scores[0], bar_width)
            
# accuracy of SVM
rects2 = plt.bar(index + bar_width, scores[1], bar_width)
                 
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, round(height,2), ha='center', va='bottom',weight="bold")
       # set the edge color of bar to white
        rect.set_edgecolor('white')
plt.grid(linestyle='--',axis='y')
add_labels(rects1)
add_labels(rects2)

plt.xticks(index + bar_width/2,subjects,fontsize=11)
plt.yticks(np.arange(0.0,1.1,step=0.1),fontsize=9)
plt.xlabel('Different encoding schemes',fontsize=11,weight="bold")
plt.ylabel('Accuracy',fontsize=11,weight="bold")
plt.rc('axes', axisbelow=True)
plt.legend(model,loc='upper center', bbox_to_anchor=(0.5, 1.25), fancybox=True, ncol=5,fontsize=11)



plt.subplot(2,1,2)
model = ('RF', 'SVM')
subjects = ('Raw data','Normaliser','Standard scaler','MinMax scaler')
scores = (rfaccte_scaleunits,svmaccte_scaleunits)


bar_width = 0.2
index = np.arange(len(scores[0]))

# accuracy of RF
rects1 = plt.bar(index, scores[0], bar_width)
            
# accuracy of SVM
rects2 = plt.bar(index + bar_width, scores[1], bar_width)
                 
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, round(height,2), ha='center', va='bottom',weight="bold")
       # set the edge color of bar to white
        rect.set_edgecolor('white')
plt.grid(linestyle='--',axis='y')
add_labels(rects1)
add_labels(rects2)

plt.xticks(index + bar_width/2,subjects,fontsize=11)
plt.yticks(np.arange(0.0,1.1,step=0.1),fontsize=9)
plt.xlabel('Different scaling schemes',fontsize=11,weight="bold")
plt.ylabel('Accuracy',fontsize=11,weight="bold")
plt.rc('axes', axisbelow=True)
plt.legend(model,loc='upper center', bbox_to_anchor=(0.5, 1.25), fancybox=True, ncol=5,fontsize=11)
plt.tight_layout()
#plt.savefig('(Database2) comparsion .jpg',dpi=300)
plt.show()

# In[]
"""
Feature importance
"""
x_train, x_test, y_train, y_test = train_test_split(feature_3point,labels_3point, test_size=0.2)
def important_feats(x_train, y_train):
        model = RandomForestClassifier()
        model.fit(x_train,y_train)
        #plot graph of feature importances for better visualization
        plt.figure(figsize=(6,3))
        feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
        feat_importances.nlargest(x_train.shape[1]).plot(kind='barh',color='b',fontsize=12)
        plt.tight_layout()
        important_feats=feat_importances.nlargest(x_train.shape[1])
#        plt.savefig('Fig.23 Feature importance of Dataset 2',dpi=300)
        return important_feats
    
    #identify the top i important features and form the new test and train split
important_feats = important_feats(x_train, y_train)
print(important_feats)

nfeature = list(range(5,13,1))
epoches = list(range(1,10))
rfte_impf=np.zeros((len(epoches),len(nfeature)))
rftr_impf=np.zeros((len(epoches),len(nfeature)))
rfcomputation_time=np.zeros((len(epoches),len(nfeature)))
for j in epoches:
    for i in nfeature:
        time_start=time.time()
        x_train, x_test, y_train, y_test = train_test_split(feature_3point,labels_3point, test_size=0.2)
        x_train_new=x_test_new=[]
        imporfeature_list = important_feats.index[0:i].tolist()
        print(imporfeature_list)
        x_train_new = x_train[imporfeature_list]
        x_test_new = x_test[imporfeature_list]
        # train model 
        tree_num=200 # Tree number
        tee_depth=8  # Tree depth
        rfr = RandomForestClassifier(n_estimators= tree_num, max_features = "auto", oob_score = True,max_depth=tee_depth,random_state=50) 
        rfr.fit(x_train_new, y_train)
        rfte_impf[epoches.index(j)][nfeature.index(i)]=rfr.score(x_test_new,y_test)
        # Training Prediction
        y_train_predict_impf= rfr.predict(x_train_new)
        count = 0
        for index in range(len(y_train_predict_impf)):
            if y_train_predict_impf[index]==y_train[index]:
                count += 1
        rftr_impf[epoches.index(j)][nfeature.index(i)]= float(count)/len(y_train_predict_impf)
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
        x_train, x_test, y_train, y_test = train_test_split(feature_3point,labels_3point, test_size=0.2)
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
        # Training Prediction
        y_train_predict_impf= model.predict(x_train_new)
        count = 0
        for index in range(len(y_train_predict_impf)):
            if y_train_predict_impf[index]==y_train[index]:
                count += 1
        svmtr_impf[epoches.index(j)][nfeature.index(i)]= float(count)/len(y_train_predict_impf)
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


fig=plt.figure(figsize=(4.5,3)) # width, height in inches
plt.plot(nfeature,rfte_mean,linewidth=4.0,marker='o' ,markersize=7)
plt.plot(nfeature,svmte_mean,linewidth=4.0,marker='o',markersize=7)
plt.xlabel('Number of features',fontsize=14,fontweight='bold')
plt.ylabel('Accuracy',fontsize=14,fontweight='bold')
plt.yticks(np.arange(0.5,0.8,step=0.05))
plt.xticks(np.arange(5,13,step=1))
plt.legend(['RF-testing','SVM-testing'],fontsize=12,ncol=2,loc='lower right')
plt.grid()
plt.tight_layout()
#plt.savefig('Fig.24 The effects of input features.jpg',dpi=300)
plt.show()


# In[]
"""
Validation curve
"""
param_range = np.array(range(1,21))
X = feature_3point
y = labels_3point
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

rfc = RandomForestClassifier(n_estimators=100)
train_scores, test_scores = validation_curve(
   rfc, X, y, param_name="max_depth", param_range=param_range,
    cv=cv, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fig4=plt.figure(figsize=(5,3))
plt.title("Validation Curve with RF")
plt.xlabel("Tree depth",fontsize=14,fontweight="bold")
plt.ylabel("Score",fontsize=14,fontweight="bold")
plt.ylim(0.5,1.01)
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
#plt.savefig('(Database II) Validation Curves (RF, tree_number=100).jpg',dpi=300)
plt.show()

# In[]
"""
Learning curve
"""
X = feature_3point
y = labels_3point
tree_depth = [1,5,10,20]
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

fig5=plt.figure(figsize=(10,4))
col=2
rows=2
tit1=['(ii.1)','(ii.2)','(ii.3)','(ii.4)']
for i in range(len(tree_depth)):
    plt.subplot(rows,col,i+1)
    plt.fill_between(train_sizes, train_scores_mean[:,i] - train_scores_std[:,i],
                     train_scores_mean[:,i] + train_scores_std[:,i], alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean[:,i] - test_scores_std[:,i],
                     test_scores_mean[:,i] + test_scores_std[:,i], alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean[:,i], 'o-', color="r",linewidth=3.0,label="Training score")
    plt.plot(train_sizes, test_scores_mean[:,i], 'o-', color="g",linewidth=3.0,label="Cross-validation score")
    plt.title("%s"%tit1[i] + "Tree depth=%d"%tree_depth[i],fontsize=13)
    plt.ylabel("Score",fontsize=12,fontweight="bold")
    #plt.ylim(0.8, 1.01)
    #plt.yticks(np.arange(0.8,1.01,step=0.05))
    plt.xticks(np.arange(1000,11000,step=2000))
    plt.grid() 
    plt.legend(ncol=2,loc="best",fontsize=10)
    plt.xlabel("Training samples",fontsize=12,fontweight="bold")
plt.tight_layout()
#plt.savefig('(DatabaseII) Learning Curves.jpg',dpi=300)
plt.show()