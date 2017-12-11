# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:37:48 2017

@author: Kara
"""

import numpy as np
from hmmlearn.hmm import MultinomialHMM


def stateProbs(stateSequence):
    zeros = 0.0
    ones = 0.0
    zo = 0.0
    oz = 0.0
    for i in range(len(stateSequence)-1):
        if (stateSequence[i], stateSequence[i+1]) == (0,0):
            zeros = zeros + 1
        if (stateSequence[i], stateSequence[i+1]) == (0,1):
            oz = oz + 1
        if (stateSequence[i], stateSequence[i+1]) == (1,0):
            zo = zo + 1
        if (stateSequence[i], stateSequence[i+1]) == (1,1):
            ones = ones + 1
    matrix = np.array([[zeros/(len(stateSequence)-1), zo/(len(stateSequence)-1)],[oz/(len(stateSequence)-1),ones/(len(stateSequence)-1)]])
    return matrix

def eProbs(eSequence, stateSequence):
    high = 0.0
    highMid = 0.0
    lowMid = 0.0
    low = 0.0
    losses = []
    wins = []
    for i in range(len(stateSequence)):
        if stateSequence[i] == 0.0:
            losses.append(eSequence[i])
        else:
            wins.append(eSequence[i])
    for percent in losses:
        if percent >= .75:
            high = high + 1
        elif percent >= .50:
            highMid = highMid + 1
        elif percent >= .25:
            lowMid = lowMid + 1
        else:
            low = low + 1
    matrix = np.zeros((2,4))
    matrix[0,0] = low/len(losses)
    matrix[0,1] = lowMid/len(losses)
    matrix[0,2] = highMid/len(losses)
    matrix[0,3] = high/len(losses)
    high = 0.0
    highMid = 0.0
    lowMid = 0.0
    low = 0.0
    for percent in wins:
        if percent >= .75:
            high = high + 1
        elif percent >= .50:
            highMid = highMid + 1
        elif percent >= .25:
            lowMid = lowMid + 1
        else:
            low = low + 1
    matrix[1,0] = low/len(wins)
    matrix[1,1] = lowMid/len(wins)
    matrix[1,2] = highMid/len(wins)
    matrix[1,3] = high/len(wins)
    return matrix


# Load Data
filename = 'data.csv'
X = np.loadtxt(filename, delimiter=',')

player1 = X[:, 0]
player2 = X[:, 1]
record = X[:,2]

print "stateProbs(record)", stateProbs(record)
print "eProbs(player1, record", eProbs(player1,record)
clf = MultinomialHMM(n_components=2)
clf.transmat_ = stateProbs(record)
clf.emissionprob_ = eProbs(player1, record)
print "here"
clf.fit(clf.transmat_, clf.emissionprob_)
clf.predict(player1)







