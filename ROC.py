import pandas as pd
import numpy as np
from scipy import integrate
import time

#import test data and prediction data
test = pd.read_csv("./data/test.csv")
mbr_pred = pd.read_csv("./data/mbr_pred.csv")

test.index = test['ECI']
mbr_pred.index = mbr_pred['ECI']

test = test.drop('ECI',1)
mbr_pred = mbr_pred.drop('ECI',1)

dat = pd.read_csv("./data/datHCC.csv")
dat.index = dat['ECI']
dat = dat.drop(['ECI','HCC_RAF','HCCCount'],1)
dat = dat.astype('float64')


############
# Calc ROC #
############

start = time.time()

K = 10000

tpr = np.zeros((K, test.shape[1]))
fpr = np.zeros((K, test.shape[1]))

for i in range(K):

	p = (np.float64(i+1) / np.float64(K))

	out = (mbr_pred >= p) * 1
	out = out.astype('float64')

	error = test - out

	tpr[i,] = np.sum(test * out) / np.sum(test)
	fpr[i,] = np.sum(error==-1) / np.sum(dat==0)

print time.time() - start


############
# Calc AUC #
############

auc = np.zeros(test.shape[1])

for i in range(test.shape[1]):
	auc[i] = -1 * integrate.trapz(tpr[:,i], fpr[:,i])

auc = np.around(auc, decimals = 3)

auc = pd.DataFrame(auc, index = test.columns.values, columns = ["AUC"])
auc.to_csv("./data/auc.csv")
