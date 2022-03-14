# naiveBayes.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]))

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        # ************
        #  Question 0
        # ************
        "*** YOUR CODE HERE ***"
        features = self.features
        labels = self.legalLabels

        # print('trainingData ', trainingData)
        # print('trainingLabels ', trainingLabels)
        # print('validationData ', validationData)
        # print('validationLabels ', validationLabels)
        # print('kgrid ', kgrid)


        comPriors = util.Counter()
        comConditProb = util.Counter()
        comCounts = util.Counter()

        trainDataLength = len(trainingData)
        temp = float('-Inf')

        for x in range(trainDataLength):
            data = trainingData[x]
            label = trainingLabels[x]
            comPriors[label] = comPriors[label] + 1
            for feature, value in data.items():
                comCounts[(feature, label)] = comCounts[(feature, label)] + 1
                if value > 0:
                    comConditProb[(feature, label)] = comConditProb[(feature, label)] + 1

        # print('comPriors ', comPriors)
        # print('comCounts ', comCounts)
        # print('comConditionProbs ', comConditProb)
        
        for k in kgrid:
            priors = util.Counter()
            conditProbs = util.Counter()
            counts = util.Counter()

            for feature, label in comPriors.items():
                priors[feature] = priors[feature] + label


            for feature, value in comCounts.items():
                counts[feature] = counts[feature] + value


            for feature, label in comConditProb.items():
                conditProbs[feature] = conditProbs[feature] + label

        
            for label in labels:
                for feature in features:
                    counts[(feature, label)] += k * 2
                    conditProbs[(feature, label)] += k

            priors.normalize()
            for f, c in conditProbs.items():
                conditProbs[f] = c * 1.0 / counts[f]


            self.prior = priors
            self.conditionalProb = conditProbs


            thoughts = self.classify(validationData)
            
            tempCount = [thoughts[i] == validationLabels[i]

                for i in range(len(validationLabels))].count(True)

            if tempCount > temp:
                tempK = k
                temCount = counts
                tempProbs = conditProbs
                tempPriors = priors
                temp = tempCount

        self.prior = tempPriors
        self.conditionalProb = tempProbs
        self.k = tempK

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()
        # ************
        #  Question 0
        # ************
        "*** YOUR CODE HERE ***"
        features = self.features
        labels = self.legalLabels
        priors = self.prior
        data = datum
        conditProbs = self.conditionalProb

        for label in labels:
            logJoint[label] = math.log(priors[label])

            for feature, value in data.items():

                if value > 0:
                    logJoint[label] = logJoint[label] + math.log(conditProbs[feature,label])
                else:
                    temp = 1 - conditProbs[feature,label]
                    logJoint[label] = logJoint[label] + math.log(temp)

        return logJoint

