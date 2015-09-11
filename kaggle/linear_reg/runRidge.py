from sklearn import linear_model
import sys
sys.path.append('../template_script')
import templateScript

ridge = linear_model.Ridge(alpha=0.000001)
templateScript.doScript("linear_reg",ridge)
