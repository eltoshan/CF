import pandas as pd
import numpy as np

#import data and do a bit of clean up
dat = pd.read_csv("./data/DAT.csv")
dat = dat.astype('float32')

colnames = dat.columns.values
colnames[0] = "eci"
dat.columns = colnames

rownames = dat['eci']
dat.index = rownames

dat = dat.drop('eci',1)

#reserve ~10% of data for test set
np.random.seed(123123)
n, p = 1, 0.1
binoms = np.random.binomial(n, p, (dat.shape[0], dat.shape[1]))

test = dat * binoms
train = dat - test
#print train[:3]
#print train.shape


####################
# helper functions #
####################

# inverse frequency vector similarity (cosine similarity)
def IFVS(a, i, InvFreq):
	w = np.sum((InvFreq * a) * (InvFreq * i)) /\
	 (np.sqrt(np.sum((InvFreq*InvFreq) * (a*a))) *\
	  np.sqrt(np.sum((InvFreq*InvFreq) * (i*i))))
	return w

# prediction score
def pred(similarities, data, cc):
	allj = data[cc]
	jbar = np.sum(allj) / train.shape[0]
	score = np.sum(allj, similarities) / np.sum(similarities)
	p = jbar + (1 - jbar) * score
	return p


###########################
# collaborative filtering #
###########################

import time

start = time.time()

# inverse frequency table
InvFreq = np.array(np.log(train.shape[0] / np.sum(train, 0))).T
InvFreq[np.isinf(InvFreq)] = 0


# similarity matrix, faster than looping through with IFVS function
similarity = np.dot(InvFreq * train, (InvFreq * train).T)
sqr_mag = np.diag(similarity)
inv_sqr_mag = 1 / sqr_mag
inv_sqr_mag[np.isinf(inv_sqr_mag)] = 0
inv_mag = np.sqrt(inv_sqr_mag)
cosine_sim = similarity * inv_mag
cosine_sim = cosine_sim.T * inv_mag

# set diagonal to 0	
np.fill_diagonal(cosine_sim,0)

print time.time() - start

print cosine_sim[:3,:3]
print IFVS(np.array(train.iloc[[2]]),np.array(train.iloc[[0]]),InvFreq)
print cosine_sim.shape


# prediction

start = time.time()

mbr_pred = np.zeros([test.shape[0], test.shape[1]])

for (mbr, j) in np.ndindex(mbr_pred.shape):
	cc = dat.columns.values[j]
	if train.iloc[mbr][j] == 1:
		mbr_pred[mbr, j] = 0
	else:
		mbr_pred[mbr, j] = pred(similarities=similarity[mbr,:], data=train, cc=cc)

print time.time() - start

print mbr_pred[:3,:10]