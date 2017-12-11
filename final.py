import numpy as np 
from sklearn.hmm import MultinomialHMM
from sklearn.neural_network import MLPClassifier

filename=  'data.csv'
X = np.loadtxt(filename, delimiter)