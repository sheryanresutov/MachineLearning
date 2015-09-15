import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd

trainFile = "../training_data/train_data.csv"
testFile = "../training_data/test_x_data.csv"
with open(trainFile,mode='rb') as file:
            train = pd.DataFrame(pd.read_csv(file))

data = train.values.tolist()
data = np.asarray(data,dtype=np.double)

X = np.delete(data, 561, 1)
y = [data[561] for data in data]

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel='linear')
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=5,
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)
print("Optimal feature rankings : " + rfecv.ranking_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()