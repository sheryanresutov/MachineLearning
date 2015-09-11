from sklearn.ensemble import GradientBoostingClassifier
import sys
sys.path.append('../template_script')
import templateScript

rf = GradientBoostingClassifier(n_estimators=200, learning_rate=1, max_features='log2')
templateScript.doScript("rf",rf)
