from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, enet_path, Lasso

trainFile = "./Data/train_data.csv"
with open(trainFile,mode='rb') as file:
            train = pd.DataFrame(pd.read_csv(file))
colNames = list(train.columns.values)
print(colNames)

X = train.values.tolist()
X = np.asarray(X,dtype=np.double)
print(X)
cv = cross_validation.KFold(len(X), n_folds=5)

for traincv, testcv in cv:
    train_set=X[traincv]
    test_set = np.delete(X[testcv],560,1)
    print(test_set)

        #Random Forest
    rf_target = [data[560] for data in train_set]
    rf_train = np.delete(train_set,560,1)
    print(rf_train)

    rf = RandomForestClassifier(n_estimators=10)
    print("trying to fit")
    rf.fit(rf_train, rf_target)
    print("trying to predict")
    rf_predicted = rf.predict(test_set)

    # for i in range(50):
    #     if abs(label_set[i] - rf_predicted[i]) < binRange:
    #             rf_count+=1
    print(rf_predicted)

#print('5-fold CrossValidated 10 Averaged Random Forests Accuracy is: '+str(float(rf_count/2500.000))+'\n')

