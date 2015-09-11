from sklearn import linear_model
import sys
sys.path.append('../template_script')
import templateScript

p = linear_model.Perceptron(alpha=0.1)
templateScript.doScript("linear_reg",p)
