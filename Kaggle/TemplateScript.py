from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, enet_path, Lasso

trainFile = "./Data/train_data.csv"
testFile = "./Data/test_x_data.csv"
with open(trainFile,mode='rb') as file:
            train = pd.DataFrame(pd.read_csv(file))

X = train.values.tolist()
X = np.asarray(X,dtype=np.double)
b = [i[561] for i in X]
b = np.asarray(b,dtype=np.double)
cv = cross_validation.KFold(len(X), n_folds=5)
rf_count=0

for traincv, testcv in cv:
    train_set=X[traincv]
    test_set = np.delete(X[testcv],561,1)
    label_set= b[testcv]
    #Random Forest
    rf_target = [data[561] for data in train_set]
    rf_train = np.delete(train_set,561,1)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(rf_train, rf_target)
    rf_predicted = rf.predict(test_set)



    for i in range(len(label_set)):
        if(label_set[i] != rf_predicted[i]):
            rf_count=rf_count+1

with open(testFile,mode='rb') as file:
    test = pd.DataFrame(pd.read_csv(file))
X_test=test.values.tolist()
X_test=np.asarray(X_test, dtype=np.double)

rf_test_predicted = rf.predict(X_test)
np.savetxt("test_y_data.csv",rf_test_predicted)

print(rf_test_predicted)
print(1-rf_count/7767.000)
