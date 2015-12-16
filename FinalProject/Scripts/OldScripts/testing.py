from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, enet_path, Lasso

print('Management Pay : 1')
print('Faculty Pay    : 2')
print('Starting Pay   : 3')
print('Median SAT     : 4')
stdDev=[24791.000,30000.000,6001.000,148.000]
feature = raw_input("Enter the feature # you want to predict: ")
feature = int(feature)
binRange = raw_input("Enter the binRange you want to test (about 20000 for the faculty/management categories, 5000 for starting, 100 for Median_SAT): ")
binRange = float(binRange)/stdDev[feature-1]

trainFile = "./totalData.csv"
with open(trainFile,mode='rb') as file:
            train = pd.DataFrame(pd.read_csv(file))
colNames = list(train.columns.values)
offset=0
#figure out which features to use
for name in list(train.columns.values):
    if(((name == 'Management_Pay') & (feature == 1)) | 
    ((name == 'Faculty_Pay') & (feature == 2)) | 
    ((name == 'Starting_Pay') & (feature == 3)) |  
    ((name == 'Median_SAT') & (feature == 4 ))):
        colNames.remove(name)
        continue;
    ans = raw_input("Would you like to use "+name+" as a feature? (y/n)")
    if ans == 'y':
        continue
    elif ans == 'n':
        del train[name]
        colNames.remove(name)
        if ((feature == 2) & (name== 'Management_Pay')):
            offset += 1
        elif ((feature == 3) & ((name== 'Management_Pay') | (name == 'Faculty_Pay'))):
            offset += 1
        elif ((feature == 4) & ((name== 'Management_Pay') | (name == 'Faculty_Pay') | (name == 'Starting_Pay'))):
            offset += 1
print('\n')
X = train.values.tolist()
X = np.asarray(X,dtype=np.double)
b = [i[feature-1-offset] for i in X]
b = np.asarray(b,dtype=np.double)

cv = cross_validation.KFold(len(X), n_folds=5)
ind = 0
rf_count=0
l_count=0
indAccuracy = [None]*5
for traincv, testcv in cv:
    #count=0
    train_set=X[traincv]
    test_set = np.delete(X[testcv],feature-1-offset,1)
    label_set= b[testcv]
        #Random Forest
    rf_target = [data[feature-1-offset] for data in train_set]
    rf_train = np.delete(train_set,feature-1-offset,1)
    rf = RandomForestClassifier(n_estimators=100)
    for it in range(10): 
        rf.fit(rf_train, rf_target)
        rf_predicted = rf.predict(test_set)
        for i in range(50):
            if abs(label_set[i] - rf_predicted[i]) < binRange:
                    rf_count+=1
        #Lasso Regression
    lasso = Lasso(alpha=.1)
    lasso_train = np.delete(train_set,feature-1-offset,1)
    lasso_target = [data[feature-1-offset] for data in train_set]
    lasso_predicted = lasso.fit(lasso_train, lasso_target).predict(test_set)
    for i in range(50):
        if abs(label_set[i] - lasso_predicted[i]) < binRange:
                l_count+=1


print('5-fold CrossValidated 10 Averaged Random Forests Accuracy is: '+str(float(rf_count/2500.000))+'\n')

print('5-fold CrossValidated Lasso Regression Accuracy is: '+str(float(l_count/250.000))+'\n')

resp = raw_input("Would you like to see the lasso plot? (y/n)")
if resp == 'y':
    eps = 5e-6  # the smaller it is the longer is the path 
    alphas_lasso, coefs_lasso, _ = lasso_path(np.delete(X,feature-1-offset,1), b, eps, fit_intercept=False)
    plt.figure(1)
    ax = plt.subplot(111)
    colormap = plt.cm.gist_ncar
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(colNames))])
    for i in range(len(coefs_lasso)):
        ax.plot(-np.log10(alphas_lasso), coefs_lasso[i].T,label=colNames[i])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    plt.title('Lasso')
    plt.axis('tight')
    plt.show()
