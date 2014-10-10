import numpy as np
import pandas as pd
import time

###############################################################################
"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + np.square(R[i][j] - np.dot(P[i,:],Q[:,j]))
                    for k in xrange(K):
                        e = e + (beta/2) * ( np.square(P[i][k]) + np.square(Q[k][j]) )
        if e < 0.001:
            break
    return P, Q.T

###############################################################################

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

###############################################################################

R = train

R = np.array(R)

N = R.shape[0]
M = R.shape[1]
K = 2

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

start = time.time()
nP, nQ = matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02)
print time.time() - start

###############################################################################

mbr_pred = np.dot(nP, nQ.T)
mbr_pred = mbr_pred/np.max(mbr_pred)
mbr_pred[R==1] = 0
