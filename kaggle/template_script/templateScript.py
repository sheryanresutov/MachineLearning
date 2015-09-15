from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import datetime
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV

trainFile = "../training_data/train_data.csv"
testFile = "../training_data/test_x_data.csv"
with open(trainFile,mode='rb') as file:
            train = pd.DataFrame(pd.read_csv(file))

X = train.values.tolist()
X = np.asarray(X,dtype=np.double)
b = [i[561] for i in X]
b = np.asarray(b,dtype=np.double)
cv = cross_validation.KFold(len(X), n_folds=5)

#XX=X
#RFECV(SVR(kernel="linear"), step=1, cv=5).fit_transform(np.delete(XX,561,1), b)

def doScript(submissionFolder,model):
    count=0
    for traincv, testcv in cv:
        train_set=X[traincv]
        test_set = np.delete(X[testcv],561,1)
        label_set= b[testcv]
        train_data = np.delete(train_set,561,1)
        target_data = [data[561] for data in train_set]
        
        model.fit(train_data, target_data)
        predicted = model.predict(test_set)
        for i in range(len(label_set)):
            if(label_set[i] != predicted[i]):
                count=count+1

    accuracy = str(1-count/7767.000)
    print("Accuracy: "+accuracy)                

    with open(testFile,mode='rb') as file:
        test = pd.DataFrame(pd.read_csv(file))
    X_test=test.values.tolist()
    X_test=np.asarray(X_test, dtype=np.double)
    model.fit(np.delete(X,561, 1), [data[561] for data in X])
    test_predicted = model.predict(X_test)
    np.savetxt("../"+submissionFolder+"/submissions/"+str(model)+"_"+accuracy+"_"+str(datetime.datetime.now().time())+".csv",test_predicted)

