#!/usr/bin/env python

import sys
import itertools
import time
from random import randint
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

def parseRating(line):
    """
    Parses a rating record for HCC condition flags in format ECI,hccID,flag .
    """
    fields = line.strip().split(",")
    return randint(0,9) % 10, \
    	(float(fields[0]), float(fields[1]), float(fields[2]))

def parseHCC(line):
    """
    Parses a HCC record in format hccID,hccTitle .
    """
    fields = line.strip().split(",")
    return int(fields[0]), fields[1]

def loadRatings(ratingsFile):
    """
    Load ratings from file.
    """
    if not isfile(ratingsFile):
        print "File %s does not exist." % ratingsFile
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
    f.close()
    if not ratings:
        print "No ratings provided."
        sys.exit(1)
    else:
        return ratings

def computeRmse(model, data, n):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))


if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "CFsparkALS.py hccDataDir"
        sys.exit(1)

    # set up environment
    conf = SparkConf() \
      .setAppName("CFsparkALS") \
      .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)

    # load ratings and HCC labels
    dataHomeDir = sys.argv[1]

    # ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
    ratings = sc.textFile(join(dataHomeDir, "datLong.csv")).map(parseRating)

    # movies is an RDD of (movieId, movieTitle)
    #HCCs = dict(sc.textFile(join(dataHomeDir, "hccLabels.dat")).map(parseHCC).collect())



    numRatings = ratings.count()
    numMbrs = ratings.values().map(lambda r: r[0]).distinct().count()
    numHCCs = ratings.values().map(lambda r: r[1]).distinct().count()

    print "Got %d ratings from %d users on %d HCCs." % (numRatings, numMbrs, numHCCs)

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

    ranks = [4, 12]
    numIters = [5, 10]
    alphas = [0.01, 1.0, 100.0]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestAlpha = -1
    bestNumIter = -1

    for rank, numIter, alph in itertools.product(ranks, numIters, alphas):
    	start = time.time()
        model = ALS.trainImplicit(training, rank, numIter, alpha = alph)
        validationRmse = computeRmse(model, validation, numValidation)
        print "Training time: %.2f seconds" % (time.time() - start)
        print "RMSE (validation) = %f for the model trained with " % validationRmse + \
              "rank = %d, alpha = %.2f, and numIter = %d." % (rank, alph, numIter)
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestAlpha = alph
            bestNumIter = numIter

    testRmse = computeRmse(bestModel, test, numTest)

    print "The best model was trained with rank = %d, alpha = %.2f " % (bestRank, bestAlpha) \
        + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRmse)


    # compare explicit vs implicit

    # clean up
    sc.stop