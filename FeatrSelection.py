# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:04:41 2019

@author: RMSOEE
"""
###################################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

##################################################################################################################
data = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Models\Model_PHQ.csv', header=0) 

## General model
data.drop(data.columns[[0,15]], axis=1, inplace=True)  
 
X_All = data.iloc[:,0:14]
sc_x=StandardScaler()
X = sc_x.fit_transform(X_All)
y = data.iloc[:,14]

data.shape
data.columns
data.head()
X.head()

data = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Models\Model_PSS.csv', header=0)
data = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Models\Model_PSQI.csv', header=0)
data = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Models\Model_Loneliness.csv', header=0)
data = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Models\Model_Flourishing.csv', header=0)

###################################################################################################################
####################### Boruta Variable Importance Algorithm ######################################################
rf = RandomForestClassifier(n_estimators=100, n_jobs=4, class_weight='balanced', max_depth=5) 
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2) 
feat_selector.fit(X, y)
print(feat_selector.ranking_)

feat_selector.n_features_
feat_selector.support_
################################## Recursive Feature Elimination ##################################################

lr = LinearRegression(normalize=True)
lr.fit(X,y)

rfe = RFE(lr, n_features_to_select=1, verbose =3 )
rfe = rfe.fit(X,y)


rfe = RFE(estimator=clf4, n_features_to_select=1, verbose =3 )
rfe = rfe.fit(X,y)

rfe = RFE(rf, n_features_to_select=1, verbose =3 )
rfe = rfe.fit(X,y)

print(rfe.support_)
print(rfe.ranking_)

###################### Recursive Feature Elimination - Cross validation ##################

lr = LinearRegression()#normalize=True)
lr.fit(X,y)

rfecv = RFECV(lr, step=1,  cv=kf,  verbose =3, scoring='roc_auc') # n_features_to_select=1,
rfecv.fit(X, y)

rfecv = RFECV(estimator=clf4, step=1,  cv=kf,  verbose =3, scoring='roc_auc') # n_features_to_select=1,
rfecv.fit(X, y)

rfecv = RFECV(rf, step=1,  cv=kf,  verbose =3, scoring='roc_auc') # n_features_to_select=1,
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)
print(rfecv.ranking_)
rfecv.support_ 


print("Optimal number of features : %d" % rfecv.n_features_)

######################################################################################################

# Using Random Forest
rf = RandomForestRegressor(n_jobs=-1, n_estimators=100, verbose=3)
rf.fit(X,y)

ranks["RF"] = ranking(rf.feature_importances_, colnames)


