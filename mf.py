import numpy as np
import pandas as pd


class ALSModel():

    def __init__(self, ratings, num_factors=20, num_iterations=10, reg_param=0.30):
        self.Q = ratings
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param

    def train_model(self):
        m = self.Q.shape[0]
        n = self.Q.shape[1]
        nRatings = np.count_nonzero(self.Q)

        W = self.Q>0.5
        W[W == True] = 1
        W[W == False] = 0
        W = W.astype(np.float64, copy=False)
        Nu = np.sum(W, axis=1)
        Ni = np.sum(W, axis=0)

        X = np.random.rand(m, self.num_factors) 
        Y = np.random.rand(self.num_factors, n)
        lambda_eye = self.reg_param * np.eye(self.num_factors)

        for ii in range(self.num_iterations):
            for u, Wu in enumerate(W):
                X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + max(1, Nu[u]) * lambda_eye,
                                       np.dot(Y, np.dot(np.diag(Wu), self.Q[u].T))).T
            for i, Wi in enumerate(W.T):
                Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + max(1, Ni[i]) * lambda_eye,
                                         np.dot(X.T, np.dot(np.diag(Wi), self.Q[:, i])))
            print "%dth iteration is completed" % (ii+1)
        return factoredModel(X, Y)


class factoredModel():

    def __init__(self, userMat, itemMat):
        self.X = userMat
        self.Y = itemMat

    def computeRMSE(self, ratings):
        Q = ratings
        nRatings = np.count_nonzero(ratings)
        W = ratings>0.5
        W[W == True] = 1
        W[W == False] = 0
        W = W.astype(np.float64, copy=False)
        Qhat = W * np.dot(self.X, self.Y)
        RMSE = np.sqrt(np.sum((W * (Q - Qhat)**2)) / nRatings)
        return Qhat, RMSE


if __name__ == "__main__":

    # load data
    ratings = pd.read_csv("./data/nrgRatings.csv", header=None)
    ratings.columns = ['climbID','userID','rating']
    dat = ratings.pivot_table(cols=['climbID'], rows=['userID'], values=['rating'])
    dat = dat.fillna(0)

    # train/test split
    np.random.seed(123123)
    n, p = 1, 0.2
    test = np.random.binomial(n, p, (dat.shape[0], dat.shape[1]))
    train = np.zeros((dat.shape[0], dat.shape[1]))
    train[test==0] = 1
    test = (test * dat).values
    train = (train * dat).values

    # create model
    model = ALSModel(train)
    trainedModel = model.train_model()

    # evaluate on test set
    Qhat, testRMSE = trainedModel.computeRMSE(test)

    print "Test set RMSE = %f" % testRMSE
