from sklearn import linear_model
import sys
sys.path.append('../template_script')
import templateScript

log = linear_model.LogisticRegression(C=1e5)
templateScript.doScript("linear_reg",log)
