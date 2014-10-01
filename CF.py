import pandas as pd
import numpy as np

#import data and do a bit of clean up
dat = pd.read_csv("./data/datHCC.csv")

dat.index = dat['ECI']
dat = dat.drop(['ECI','HCC_RAF','HCCCount'],1)
dat = dat.astype('float32')

#reserve ~10% of data for test set
np.random.seed(123123)
n, p = 1, 0.1
binoms = np.random.binomial(n, p, (dat.shape[0], dat.shape[1]))

test = dat * binoms
train = dat - test


# print train[:3]
# print train.shape
# print np.sum(np.sum(dat))
# print np.sum(np.sum(train))
# print np.sum(np.sum(test))
# exit()

####################
# helper functions #
####################

# inverse frequency vector similarity (cosine similarity)
def IFVS(a, i, InvFreq):
	w = np.dot( (InvFreq * a), (InvFreq * i).T ) /\
	 (np.sqrt(np.dot(np.square(InvFreq), np.square(a).T)) *\
	  np.sqrt(np.dot(np.square(InvFreq), np.square(i).T)))
	return w

# similarity matrix, faster than looping through with IFVS function
def IFVS_mat(data, InvFreq):
	similarity = np.dot(InvFreq * data, (InvFreq * data).T)
	sqr_mag = np.diag(similarity)
	inv_sqr_mag = 1 / sqr_mag
	inv_sqr_mag[np.isinf(inv_sqr_mag)] = 0
	inv_mag = np.sqrt(inv_sqr_mag)
	cosine_sim = similarity * inv_mag
	cosine_sim = cosine_sim.T * inv_mag
	return cosine_sim

# prediction score
def pred(similarities, data, cc):
	allj = data[cc]
	jbar = np.sum(allj) / data.shape[0]
	score = np.dot(allj, similarities.T) / np.sum(similarities)
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
cosine_sim = IFVS_mat(data=train, InvFreq=InvFreq)

# set diagonal to 0	
np.fill_diagonal(cosine_sim,0)
	
print time.time() - start

print cosine_sim[20:40,:5]
print np.sum(np.sum(cosine_sim))
print IFVS(np.array(train.iloc[[38]]),np.array(train.iloc[[2]]),InvFreq)
print cosine_sim.shape




# prediction

start = time.time()

mbr_pred = np.zeros([test.shape[0], test.shape[1]])

for (mbr, j) in np.ndindex(mbr_pred.shape):
	cc = dat.columns.values[j]
	if train.iloc[mbr][j] == 1:
		mbr_pred[mbr, j] = 0
	else:
		mbr_pred[mbr, j] = pred(similarities=cosine_sim[mbr,:], data=train, cc=cc)

print time.time() - start

#print mbr_pred[:5,:5]

mbr_pred = pd.DataFrame(mbr_pred, index = test.index, columns = test.columns.values)
mbr_pred.to_csv("./data/mbr_pred.csv")
test.to_csv("./data/test.csv")