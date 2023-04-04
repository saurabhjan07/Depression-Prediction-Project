# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:04:41 2019

@author: RMSOEE
"""
#############################################################################################################################
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
#############################################################################################################################
data = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Models\Model_PHQ.csv', header=0) 
data = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Models\Model_PSS.csv', header=0)
data = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Models\Model_PSQI.csv', header=0)
data = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Models\Model_Loneliness.csv', header=0)
data = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Models\Model_Flourishing.csv', header=0)
#############################################################################################################################

## General model
data.drop(data.columns[[0,15]], axis=1, inplace=True)   
X_All = data.iloc[:,0:14]
sc_x=StandardScaler()
X = sc_x.fit_transform(X_All)
y = data.iloc[:,14]
colnames = data.columns

#################################################################################################
#data.shape
#data.columns
#data.head()
#X.head()
#############################################################################################################################

# Define dictionary to store our rankings
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


# Finally let's run our Selection Stability method with Randomized Lasso
#rlasso = RandomizedLasso(alpha=0.04)
#rlasso.fit(X, y)
#ranks["rlasso/Stability"] = ranking(np.abs(rlasso.scores_), colnames)


# Construct our Linear Regression model
lr = LinearRegression(normalize=True)
lr.fit(X,y)
#stop the search when only the last feature is left - Recursive Feature Elimination
rfe = RFE(lr, n_features_to_select=1, verbose =3 )
rfe.fit(X,y)
ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)


# Using Linear Regression
lr = LinearRegression(normalize=True)
lr.fit(X,y)
ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)

# Using Ridge 
ridge = Ridge(alpha = 7)
ridge.fit(X,y)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)

# Using Lasso
lasso = Lasso(alpha=.05)
lasso.fit(X, y)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)

# Using Random Forest Rgeressor
rf = RandomForestRegressor(n_jobs=-1, n_estimators=100, verbose=3)
rf.fit(X,y)
ranks["RF"] = ranking(rf.feature_importances_, colnames)


# Create empty dictionary to store the mean value calculated from all the scores
r = {}
try:
    for name in colnames:
        r[name] = np.mean([ranks[method][name] for method in ranks.keys()])
        #r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)
except KeyError:
    check=None
    
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
 
print("\t%s" % "\t".join(methods))
try:
    for name in colnames:
        print("%s\t%s" % (name, "\t".join(map(str, 
                         [ranks[method][name] for method in methods]))))
except KeyError:
    check=None

# Put the mean scores into a Pandas dataframe
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])

# Sort the dataframe
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)

# Let's plot the ranking of the features
sns.set(font_scale=1.5)
fig = sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", 
               size=8, aspect=1.9, palette='coolwarm')

fig.savefig("D:\OneDrive\Depression Prediction\Prediction Models\Draft Submission\Revision Feb 2020\JAIHC\Results and Codes\Feature Selection\Flourishing_600dpi.png", dpi=600)
fig.savefig("D:\OneDrive\Depression Prediction\Prediction Models\Draft Submission\Revision Feb 2020\JAIHC\Results and Codes\Feature Selection\Flourishing_300dpi.png", dpi=300)


