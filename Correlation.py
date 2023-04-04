# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:29:32 2019

@author: Saurabh
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import os
import re
import csv

df = pd.read_csv('E:\Depression Prediction\Prediction Models\Results & Others\SurveyScores.csv', header=0)
df.head
df.columns
df.shape
df.drop(df.columns[[0]], axis=1, inplace=True)


df1 = df.corr(method ='pearson')
df2 = calculate_pvalues(df)

base_filename = 'Corr1.txt'
with open(os.path.join('E:\Depression Prediction\Prediction Models', base_filename),'w') as outfile:
    df1.to_string(outfile)


base_filename = 'pval1.txt'
with open(os.path.join('E:\Depression Prediction\Prediction Models', base_filename),'w') as outfile:
    df2.to_string(outfile)
    

df = pd.read_csv('E:\Depression Prediction\Prediction Models\Results & Others\FinalDataSheet.csv', header=0)
df.head
df.columns
df.shape
df.drop(df.columns[[0]], axis=1, inplace=True)


df1 = df.corr(method ='pearson')
df2 = calculate_pvalues(df)

base_filename = 'Corr_All.txt'
with open(os.path.join('E:\Depression Prediction\Prediction Models', base_filename),'w') as outfile:
    df1.to_string(outfile)


base_filename = 'pval_All.txt'
with open(os.path.join('E:\Depression Prediction\Prediction Models', base_filename),'w') as outfile:
    df2.to_string(outfile)



df = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Results & Others\Workfile2.csv', header=0)
k = df.shape[1]
i=0
for i in range(0,k):
    for j in range(i+1,k):
        coef, p = spearmanr(df.iloc[:, i], df.iloc[:, j])
        print(df.columns[i], "vs", df.columns[j], ":", '%.3f ' % coef, ":", '%.3f ' % p )
        #print('%.3f ' % coef, " ")
        #print('%.3f ' % p, " " )
        
        #coef, p = spearmanr(data1, data2)

k = 20
l = 30
for j in range(20,l):
    print(df.columns[j])
    for i in range(0,k):
        coef, p = spearmanr(df.iloc[:, i], df.iloc[:, j])
        print('%.3f ' % coef, ":", '%.3f ' % p )
        #print('%.3f ' % coef, " ")
        #print('%.3f ' % p, " " )


df = pd.read_csv('D:\OneDrive\Depression Prediction\Prediction Models\Results & Others\PA_Final_3.csv', header=0)
k = df.shape[1]
i=0
for i in range(0,k):
    for j in range(i+1,k):
        coef, p = spearmanr(df.iloc[:, i], df.iloc[:, j])
        print(df.columns[i], "vs", df.columns[j], ":", '%.3f ' % coef, ":", '%.3f ' % p )
        #print('%.3f ' % coef, " ")
        #print('%.3f ' % p, " " )
        
        #coef, p = spearmanr(data1, data2)