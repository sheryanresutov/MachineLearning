from sklearn import linear_model
import sys
sys.path.append('../template_script')
import templateScript

p = linear_model.Perceptron(alpha=1e5)
templateScript.doScript("linear_reg",p)
