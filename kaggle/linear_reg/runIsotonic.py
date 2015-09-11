from sklearn import isotonic
import sys
sys.path.append('../template_script')
import templateScript

ir = isotonic.IsotonicRegression()
templateScript.doScript("linear_reg",ir)
