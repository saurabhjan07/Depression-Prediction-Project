# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 15:34:39 2018

@author: Saurabh
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

from math import exp, expm1,log
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


#from imblearn import under_sampling, over_sampling
###############           IMPORT Classifiers                ###################

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from scipy.stats import pearsonr

####################################################################################################################
data = pd.read_csv('D:\OneDrive\PHD Work\Depression Prediction\Prediction Models\Models\Model_PHQ.csv', header=0) 


data = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Models\Model_PSS.csv', header=0)


data = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Models\Model_PSQI.csv', header=0)

data = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Models\Model_Loneliness.csv', header=0)

data = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Models\Model_Flourishing.csv', header=0)

###############################################################################
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data['Class'].value_counts()
sns.countplot(x='Class', data = data, palette = 'hls')
#plt.savefig('count_plot')
plt.show()
data.groupby('Class').mean()

#Check the independence between the independent variables
sns.heatmap(data.corr())
#plt.savefig('heat_map')
plt.show()

data.corr(method ='pearson')

###############################################################################
data.drop(data.columns[[0,15]], axis=1, inplace=True)

X_All = data.iloc[:,0:14]
y = data.iloc[:,14]

#data.shape
#data.columns
#data.head()
#data.drop(data.columns[[1,3,4,5,8]], axis=1, inplace=True)
#data.drop(data.columns[[4,5,6]], axis=1, inplace=True)
#X_All = data.iloc[:,0:6]
#y = data.iloc[:,6]

# Data Normalization ............................................................................
sc_x=StandardScaler()
X = sc_x.fit_transform(X_All)

# Create classification model Objects ...............................................................................
clf1 = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
clf2 = KNeighborsClassifier(n_neighbors=3)
clf3 = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='auto', probability = True)
clf4 = RandomForestClassifier(max_depth=2, random_state=0)
clf5 = LogisticRegression(random_state=0)
clf6 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
sm = SMOTE(random_state=42)

#### Stratified K-Fold Cross Validation ###############################################

kf = StratifiedKFold(n_splits=5, shuffle = True, random_state=45) 

RocAuc_score =[]
Recall_score =[]
Precision_score =[]
Accuracy_score =[]


for train_index,test_index in kf.split(X,y):
    #print('{} of KFold {}'.format(i,kf.n_splits))
    
    X_train1, X_test = X[train_index],X[test_index]
    y_train1, y_test = y[train_index],y[test_index]   
    X_train, y_train = sm.fit_sample(X_train1, y_train1)        
    clf1.fit(X_train, y_train)
    ypred = clf1.predict(X_test)
    RocAuc = roc_auc_score(y_test, ypred)
    Recall = recall_score(y_test, ypred, pos_label=1, average='binary')
    Precision = precision_score(y_test, ypred, pos_label=1, average='binary')
    Accuracy = accuracy_score(y_test, ypred)
    RocAuc_score.append(RocAuc)
    Recall_score.append(Recall)
    Precision_score.append(Precision)
    Accuracy_score.append(Accuracy)

print('\nMean Precision: ', np.mean(Precision_score), '\nMean Std.: ', np.std(Precision_score))
print('Mean Recall: ', np.mean(Recall_score),  '\nMean Std.: ', np.std(Recall_score))
print('Mean Accuracy: ',np.mean(Accuracy_score), '\nMean Std.: ', np.std(Accuracy_score))
print('Mean ROC-AUC: ',np.mean(RocAuc_score), '\nMean Std.: ', np.std(RocAuc_score))


############################ Analytical Model Logistic Regression ####################################################
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

data = pd.read_csv('D:\OneDrive\PHD Work\Depression Prediction\Prediction Models\Models\Model_PHQ.csv', header=0) 
data.drop(data.columns[[0,15]], axis=1, inplace=True)

data['Class'].value_counts()


X = data.iloc[:,0:14]
y = data.iloc[:,14]

sc_x=StandardScaler()
X = sc_x.fit_transform(X_All)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

logit_model=sm.Logit(y_train, X_train)
result=logit_model.fit()
print(result.summary2())

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
logreg.get_params()
logreg.coef_
logreg.intercept_
logreg.n_iter_

#model = LogisticRegression(random_state=0)
