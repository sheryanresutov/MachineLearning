from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.append('/home/sheryan/Documents/EEclasses/Machine Learning/MachineLearning/kaggle/template_script')
import templateScript

knn = KNeighborsClassifier()
templateScript.doScript("knn",knn)