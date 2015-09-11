from sklearn.kernel_ridge import KernelRidge
import sys
sys.path.append('../template_script')
import templateScript

kr = KernelRidge(alpha=0.1)
templateScript.doScript("linear_reg",kr)
