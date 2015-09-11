from sklearn import linear_model
import sys
sys.path.append('../template_script')
import templateScript

tsr = linear_model.TheilSenRegressor()
templateScript.doScript("linear_reg",tsr)
