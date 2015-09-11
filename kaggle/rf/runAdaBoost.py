from sklearn.ensemble import AdaBoostClassifier
import sys
sys.path.append('../template_script')
import templateScript

rf = AdaBoostClassifier(n_estimators=200, learning_rate=1)
templateScript.doScript("rf",rf)
