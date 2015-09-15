from sklearn.svm import SVC
import sys
sys.path.append('/home/sheryan/Documents/EEclasses/Machine Learning/MachineLearning/kaggle/template_script')
import templateScript

svm = clf=SVC(C=19, kernel='rbf', gamma=0.019)
templateScript.doScript("svm",svm)