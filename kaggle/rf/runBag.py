from sklearn.ensemble import BaggingClassifier
import sys
sys.path.append('../template_script')
import templateScript

rf = BaggingClassifier(n_estimators=200, learning_rate=1, max_features='log2', bootstrap='False')
templateScript.doScript("rf",rf)
