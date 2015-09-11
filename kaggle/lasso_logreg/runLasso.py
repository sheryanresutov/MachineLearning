from sklearn.linear_model import lasso_path, enet_path, Lasso
import sys
sys.path.append('../template_script')
import templateScript

lasso = Lasso(alpha=0.1)
templateScript.doScript("lasso_logreg",lasso)
