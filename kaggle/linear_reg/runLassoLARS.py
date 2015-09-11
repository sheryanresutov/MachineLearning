from sklearn import linear_model
import sys
sys.path.append('../template_script')
import templateScript

lasso = linear_model.LassoLars(alpha=0.1)
templateScript.doScript("linear_reg", lasso)
