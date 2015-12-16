from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
import csv

err = []
trainFile = "./PredictManagementPay/train_set.csv"
testFile = "./PredictManagementPay/test_set.csv"
with open(trainFile,mode='rb') as file:
    		train = pd.DataFrame(pd.read_csv(file))
y_train = train["Management"].tolist()
del train["Management"]
X_train = train.values.tolist()

with open(testFile,mode='rb') as file:
    		test = pd.DataFrame(pd.read_csv(file))
X_test = test.values.tolist()

alpha = .1
lasso = Lasso(alpha=alpha, normalize=True)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
print(X_train)
print(y_train)
#err = y_pred_lasso-y_test
#count=0
#for val in err:
#	if(val<1):
#		count+=1
print (lasso.coef_)
writer = csv.writer(open('./PredictManagementPay/lasso_preds.csv', 'wb'))
for x in y_pred_lasso:
    writer.writerow([x])
#print(count*4)
#print lasso
#print "r^2 on test data : %f" % (1 - np.linalg.norm(y_test - y_pred_lasso) ** 2
#                                      / np.linalg.norm(y_test) ** 2)
