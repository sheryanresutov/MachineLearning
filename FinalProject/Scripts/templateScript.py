from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import preprocessing
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import datetime
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.svm import SVC
from sklearn import linear_model
import sys
import sqlite3 

def get_query(attributes = [], year = ""):
    query = "SELECT "
    if len(attributes) > 0:
        query += ",".join(attributes)
    else:
        query += "*"
    query += " FROM Scorecard"
    if len(attributes) > 0 and len(year) > 0:
        query += " WHERE"
        query += " year = " + year
        for attr in attributes:
            query += " AND " + attr + " != 'PrivacySuppressed' AND " + attr + " IS NOT NULL"
    return query

def get_binner(num_bins, min_val, max_val):
    def binner(val):
        rng = max_val - min_val
        bin_sz = rng / float(num_bins)
        for i in range(num_bins):
            if (i*bin_sz + min_val) > val:
                return i
        return num_bins
    return binner


def doScript(model, normalize, df,depVar):
    count=0
    for traincv, testcv in cv:
        trainSet = df.iloc[traincv]
        testSet = df.iloc[testcv]

        print(trainSet.columns[depVar])

        trainSetLabels = trainSet[trainSet.columns[depVar]]
        # NEED TO SPECIFY NUM BINS
        num_bins = 2
        binner = get_binner(num_bins, df[df.columns[depVar]].min(), df[df.columns[depVar]].max())
        trainSetLabels = trainSetLabels.apply(binner)
        trainSetAtts = trainSet.drop(trainSet.columns[depVar], axis=1) 
        testSetLabels = testSet[testSet.columns[depVar]]
        testSetLabels = testSetLabels.apply(binner)

        testSetLabels = testSetLabels.values.tolist()
        testSetAtts = testSet.drop(testSet.columns[depVar], axis=1) 
        if normalize == True:
            model.fit(preprocessing.StandardScaler().fit_transform(trainSetAtts), trainSetLabels)
            predicted = model.predict(preprocessing.StandardScaler().fit_transform(testSetAtts))
        else:
            model.fit(trainSetAtts,trainSetLabels)
            predicted = model.predict(testSetAtts)
        for i in range(len(testSetLabels)):
            if testSetLabels[i] != predicted[i]:
                count=count+1
    accuracy = str(1-count/float(len(df.index)))
    print("Train Accuracy is: "+accuracy)                


con = sqlite3.connect('../Data/database.sqlite')
attributes = raw_input("Enter attributes separated by space: ")

query = get_query(attributes.split(' '), '2011')
depVar = raw_input("Enter target: ")

train = pd.read_sql(query,con)
print("train size: "+str(len(train.index)))
depVarIndex = train.columns.get_loc(depVar)

X = train.values.tolist()
X = np.asarray(X)
cv = cross_validation.KFold(len(X), n_folds=5)
modelType = raw_input("Which model would you like to apply? (SVM, RF, LogReg) ")
# model = SVC(C=int(1), kernel="rbf", gamma=float(0.1))
if modelType == "SVM":
    params=raw_input("Enter Parameters in this order (1 rbf 0.1 3): C Kernel Gamma Degree ")
    params = params.split(" ")
    model = SVC(C=int(params[0]), kernel=params[1], gamma=float(params[2]), degree=int(params[3]))
elif modelType == "RF":
    params=raw_input("Enter Parameters in this order (200 log2): NumOfEstimators MaxFeatures ")
    params = params.split(" ")
    model = RandomForestClassifier(n_estimators=int(params[0]), max_features=params[1])   
elif modelType == "LogReg":
    params=raw_input("Enter Parameters in this order (100000): C ")
    params = params.split(" ")
    model = linear_model.LogisticRegression(C=int(params[0]))
else:
    sys.exit()
doScript(model,False,train,depVarIndex)