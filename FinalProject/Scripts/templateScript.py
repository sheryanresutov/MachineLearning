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

def doScript(model, normalize, df,depVar):
    count=0
    for traincv, testcv in cv:
        trainSet = df.iloc[traincv]
        testSet = df.iloc[testcv]

        trainSetLabels = trainSet[trainSet.columns[depVar]] 
        trainSetAtts = trainSet.drop(trainSet.columns[depVar], axis=1) 
        testSetLabels = testSet[testSet.columns[depVar]] 
        testSetLabels = testSetLabels.values.tolist()
        testSetAtts = testSet.drop(testSet.columns[depVar], axis=1) 
        
    
        if normalize == True:
            model.fit(preprocessing.StandardScaler().fit_transform(trainSetAtts), trainSetLabels)
            predicted = model.predict(preprocessing.StandardScaler().fit_transform(testSetAtts))
        else:
            model.fit(trainSetAtts,trainSetLabels)
            predicted = model.predict(testSetAtts)
        for i in range(len(testSetLabels)):
            if(testSetLabels[i] != predicted[i]):
                count=count+1
    accuracy = str(1-count/float(df.size))
    print("Train Accuracy is: "+accuracy)                

indicies = raw_input("Enter variables (Ex. 0 5 13): ")
allCols = map(int, indicies.split())

depVar = input("Which is depend (Ex. 0 5 13): ")

trainFile = "../Data/MERGED2013_PP.csv"
with open(trainFile,mode='rb') as file:
    train = pd.DataFrame(pd.read_csv(file))
    train = train[train.columns[allCols]].dropna()
    X = train.values.tolist()
    X = np.asarray(X)
    cv = cross_validation.KFold(len(X), n_folds=5)
    modelType = raw_input("Which model would you like to apply? (SVM, RF, LogReg) ")
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
    doScript(model,False,train,allCols.index(depVar))


