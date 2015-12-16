"""
=====================
Lasso and Elastic Net
=====================

Lasso and elastic net (L1 and L2 penalisation) implemented using a
coordinate descent.

The coefficients can be forced to be positive.
"""
print(__doc__)

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

trainFile = "./lassotest.csv"
with open(trainFile,mode='rb') as file:
    		train = pd.DataFrame(pd.read_csv(file))
colNames = list(train.columns.values)
colNames = colNames[1:]
y = train["Median_SAT"].tolist()
del train["Median_SAT"]
X = train.values.tolist()
X = np.asarray(X,dtype=np.double)
y = np.asarray(y,dtype=np.double)
X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)

eps = 5e-6  # the smaller it is the longer is the path

print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)

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
