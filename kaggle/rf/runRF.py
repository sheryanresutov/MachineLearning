from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('/home/sheryan/Documents/EEclasses/Machine Learning/MachineLearning/kaggle/template_script')
import templateScript

rf = RandomForestClassifier(n_estimators=200, max_features='log2', bootstrap=False)
templateScript.doScript("rf",rf)