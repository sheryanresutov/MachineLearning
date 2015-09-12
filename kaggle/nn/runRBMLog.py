from sklearn import linear_model
from sklearn.nerural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

import sys
sys.path.append('../template_script')
import templateScript

log = linear_model.LogisticRegression(C=1e5)
rbm = BernoulliRBM(random_state=0, verbose=True)


#templateScript.doScript("linear_reg",log)
