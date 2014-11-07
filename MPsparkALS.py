#!/usr/bin/env python

import sys
import itertools
import time
import random
from math import sqrt
from operator import add

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

def parseRating(line):
    """
    Parses a rating record for climb in format climbID,userID,rating .
    """
    fields = line.strip().split(",")
    return random.randint(0,9), \
    	(int(fields[1]), int(fields[0]), float(fields[2]))

def parseClimb(line):
    """
    Parses a route record in format climbID,climbName,climbGrade .
    """
    fields = line.strip().split(",")
    return int(fields[0]), str(fields[1]) + " " + str(fields[2]) 

def compareRatings(model, data):
    """
    Compare predicted ratings to actual ratings
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatingsRDD = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2])))
    return predictionsAndRatingsRDD

def computeRmse(model, data, n):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    predictionsAndRatings = compareRatings(model, data).values()
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))


if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "MPsparkALS.py ./data/nrgRatings.csv ./data/nrgClimbs.csv"
        sys.exit(1)

    # set up environment
    conf = SparkConf() \
      .setAppName("MPsparkALS") \
      .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)

    random.seed(123123)

    # ratings is an RDD of (last digit of timestamp, (userId, climbID, rating))
    ratings = sc.textFile(sys.argv[1]).map(parseRating)

    # movies is an RDD of (climbId, climbName)
    routes = dict(sc.textFile(sys.argv[2]).map(parseClimb).collect())

    numRatings = ratings.count()
    numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
    numRoutes = ratings.values().map(lambda r: r[1]).distinct().count()

    print "Got %d ratings from %d users on %d routes." % (numRatings, numUsers, numRoutes)

    # split to training/validation/test
    numPartitions = 4
    training = ratings.filter(lambda x: x[0] < 6) \
        .values() \
        .repartition(numPartitions) \
        .cache()

    validation = ratings.filter(lambda x: x[0] >= 6 and x[0] < 8) \
        .values() \
        .repartition(numPartitions) \
        .cache()

    test = ratings.filter(lambda x: x[0] >= 8).values().cache()

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    print "Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest)

    # compare models

    ranks = [4,40]
    numIters = [10]
    lambdas = [0.01,0.10]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestLambda = -1
    bestNumIter = -1

    for rank, numIter, lmbda in itertools.product(ranks, numIters, lambdas):
    	start = time.time()
        model = ALS.train(training, rank, numIter, lmbda)
        validationRmse = computeRmse(model, validation, numValidation)
        print "Training time: %.2f seconds" % (time.time() - start)
        print "RMSE (validation) = %f for the model trained with " % validationRmse + \
              "rank = %d, lambda = %.2f, and numIter = %d." % (rank, lmbda, numIter)
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter

    testRmse = computeRmse(bestModel, test, numTest)
 	
    print "The best model was trained with rank = %d, lambda = %.2f " % (bestRank, bestLambda) \
        + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRmse)

    users = ratings.values().map(lambda x: x[0]).distinct().collect()
    # user = users[0]
    user = 109673281
    userRatings = ratings.values().filter(lambda x: x[0] == user).collect()
    userClimbIDs = set(x[1] for x in userRatings)
    userCandidates = sc.parallelize([r for r in routes if r not in userClimbIDs])
    predictions = bestModel.predictAll(userCandidates.map(lambda x: (user, x))).collect()
    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:20]

    print "Routes recommended for user %d:" % user
    for i in xrange(len(recommendations)):
        print ("%2d: %s" % (i + 1, routes[recommendations[i][1]])).encode('ascii', 'ignore')

    # clean up
    sc.stop