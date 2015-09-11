from sklearn import naive_bayes
import sys
sys.path.append('../template_script')
import templateScript

nb = naive_bayes.MultinomialNB(alpha=1.0)
templateScript.doScript("linear_reg",nb)
