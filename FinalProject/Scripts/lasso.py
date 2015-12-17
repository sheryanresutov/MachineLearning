import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets
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

con = sqlite3.connect('../Data/database.sqlite')
attributes = raw_input("Enter attributes separated by space: ")
colNames = attributes.split(' ')
query = get_query(colNames, '2011') # edit year
depVar = raw_input("Enter target: ")
df = pd.read_sql(query,con)
print("df size: "+str(len(df.index)))
depVarIndex = df.columns.get_loc(depVar)
y = df[df.columns[depVarIndex]]
x = df.drop(df.columns[depVarIndex], axis=1)
X = x.values.tolist()
X = np.asarray(X, dtype=np.double)
y = np.asarray(y, dtype=np.double)
X /= X.std(axis=0) # Standardize data (easier to set the l1_ratio parameter)

eps = 5e-6 # the smaller it is the longer is the path

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
