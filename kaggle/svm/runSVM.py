from sklearn.svm import SVC
import sys
sys.path.append('/home/sheryan/Documents/EEclasses/Machine Learning/MachineLearning/kaggle/template_script')
import templateScript

svm = clf=SVC(C=16, kernel='rbf', gamma=0.02)
templateScript.doScript("svm",svm)