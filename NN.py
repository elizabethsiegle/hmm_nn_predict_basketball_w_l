# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 21:43:10 2017

@author: Kara
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

afpr = []
atpr = []
aroc_auc = []

# Load Data
filename = 'data1.csv'
X1 = np.loadtxt(filename, delimiter=',')

filename = 'data1test.csv'
X1test = np.loadtxt(filename, delimiter=',')
filename = 'data2.csv'
X2 = np.loadtxt(filename, delimiter=',')
filename = 'data2test.csv'
X2test = np.loadtxt(filename, delimiter=',')

player1 = X1[:, 0]
player2 = X1[:, 1]
both = X1[:, :2]
record = X1[:,2]

#team = X1[:,3]
player1test = X1test[:, 0]
player2test = X1test[:, 1]
bothtest = X1test[:, :2]
#teamtest = X1test[:,3]
recordtest = X1test[:,2]

# add 1s column
n = len(X1)
player1 = np.c_[np.ones([n, 1]), player1]
player2 = np.c_[np.ones([n, 1]), player2]
both = np.c_[np.ones([n, 1]), both]
# add 1s column
n = len(X1test)
player1test = np.c_[np.ones([n, 1]), player1test]
player2test = np.c_[np.ones([n, 1]), player2test]
bothtest = np.c_[np.ones([n, 1]), bothtest]


#parameters = {'max_iter': [25,50,75,100,125,150,200, 300], 'activation':['identity'], 'learning_rate_init': [.3,.2,.1,.05,.025,.01], 'hidden_layer_sizes': [(10,), (15,),(20,),(25,)]}
#svc = MLPClassifier()
#clf = GridSearchCV(svc, parameters, cv = 10,scoring = 'f1')
clf = MLPClassifier(activation = 'identity', max_iter = 125, learning_rate_init= 0.1, hidden_layer_sizes = (10,))
clf.fit(player1, record)
#print clf.best_params_
#print clf.best_score_
print clf.predict(player1test)
print ("Test accuracy: " + str(clf.score(player1test, recordtest)))
y_score = clf.predict_proba(player1test)
fpr, tpr, _ = roc_curve(recordtest, y_score[:,1])
roc_auc = auc(fpr, tpr)
afpr.append(fpr)
atpr.append(tpr)
aroc_auc.append(roc_auc)


#parameters = {'max_iter': [25,50,75,100,125,150,200, 300], 'activation':['identity'], 'learning_rate_init': [.3,.2,.1,.05,.025,.01], 'hidden_layer_sizes': [(10,), (15,),(20,),(25,)]}
#svc = MLPClassifier()
#clf2 = GridSearchCV(svc, parameters, cv = 10, scoring = 'f1')
clf2 = MLPClassifier(activation= 'identity' ,max_iter= 50, learning_rate_init= 0.3, hidden_layer_sizes= (10,))
clf2.fit(player2, record)
#print clf2.best_params_
#print clf2.best_score_
print clf2.predict(player2test)
print ("Test accuracy: " + str(clf2.score(player2test, recordtest)))
y_score = clf2.predict_proba(player2test)
fpr, tpr, _ = roc_curve(recordtest, y_score[:,1])
roc_auc = auc(fpr, tpr)
afpr.append(fpr)
atpr.append(tpr)
aroc_auc.append(roc_auc)


#parameters = {'max_iter': [25,50,75,100,125,150,200, 300], 'activation':['identity'], 'learning_rate_init': [.3,.2,.1,.05,.025,.01], 'hidden_layer_sizes': [(10,), (15,),(20,),(25,)]}
#svc = MLPClassifier()
#clf3 = GridSearchCV(svc, parameters, cv = 10, scoring = 'f1')
clf3 = MLPClassifier(activation= 'identity', max_iter= 125, learning_rate_init= 0.2, hidden_layer_sizes= (10,))
clf3.fit(both, record)
#print clf3.best_params_
#print clf3.best_score_
pred = clf3.predict(bothtest)
print ("Test accuracy: " + str(clf3.score(bothtest, recordtest)))
print pred
Acc = metrics.accuracy_score(recordtest, pred)
Recall = metrics.recall_score(recordtest, pred)
Prec = metrics.precision_score(recordtest, pred)
y_score = clf3.predict_proba(bothtest)
fpr, tpr, _ = roc_curve(recordtest, y_score[:,1])
roc_auc = auc(fpr, tpr)
afpr.append(fpr)
atpr.append(tpr)
aroc_auc.append(roc_auc)
print pred
print Acc
print Recall
print Prec


#parameters = {'max_iter': [25,50,75,100,125,150,200, 300], 'activation':['identity'], 'learning_rate_init': [.3,.2,.1,.05,.025,.01], 'hidden_layer_sizes': [(10,), (15,),(20,),(25,)]}
#svc = MLPClassifier()
#clf4 = GridSearchCV(svc, parameters, cv = 10, scoring = 'f1')
##clf4 = MLPClassifier(activation= 'identity', max_iter= 125, learning_rate_init= 0.2, hidden_layer_sizes= (10,))
#clf4.fit(team, record)
##print clf4.best_params_
##print clf4.best_score_
#pred = clf4.predict(teamtest)
#print ("Test accuracy: " + str(clf4.score(teamtest, recordtest)))
#print pred
#Acc = metrics.accuracy_score(recordtest, pred)
#Recall = metrics.recall_score(recordtest, pred)
#Prec = metrics.precision_score(recordtest, pred)
#y_score = clf4.predict_proba(bothtest)
#fpr, tpr, _ = roc_curve(recordtest, y_score[:,1])
#roc_auc = auc(fpr, tpr)
#afpr.append(fpr)
#atpr.append(tpr)
#aroc_auc.append(roc_auc)
#print pred
#print Acc
#print Recall
#print Prec



player1 = X2[:, 0]
player2 = X2[:, 1]
both = X2[:, :2]
record = X2[:,2]
team = X2[:,3]
player1test = X2test[:, 0]
player2test = X2test[:, 1]
bothtest = X2test[:, :2]
recordtest = X2test[:,2]
teamtest = X2test[:,3]
print player1
print team

# add 1s column
n = len(X2)
player1 = np.c_[np.ones([n, 1]), player1]
player2 = np.c_[np.ones([n, 1]), player2]
both = np.c_[np.ones([n, 1]), both]
# add 1s column
n = len(X2test)
player1test = np.c_[np.ones([n, 1]), player1test]
player2test = np.c_[np.ones([n, 1]), player2test]
bothtest = np.c_[np.ones([n, 1]), bothtest]


parameters = {'max_iter': [25,50,75,100,125,150,200, 300], 'activation':['identity'], 'learning_rate_init': [.3,.2,.1,.05,.025,.01], 'hidden_layer_sizes': [(10,), (15,),(20,),(25,)]}
svc = MLPClassifier()
clf = GridSearchCV(svc, parameters, cv = 10, scoring = 'accuracy')
clf.fit(player1, record)
print clf.best_params_
print clf.best_score_
print clf.predict(player1test)
print ("Test accuracy: " + str(clf.score(player1test, recordtest)))
y_score = clf.predict_proba(player1test)
fpr, tpr, _ = roc_curve(recordtest, y_score[:,1])
roc_auc = auc(fpr, tpr)
afpr.append(fpr)
atpr.append(tpr)
aroc_auc.append(roc_auc)


parameters = {'max_iter': [25,50,75,100,125,150,200, 300], 'activation':['identity'], 'learning_rate_init': [.3,.2,.1,.05,.025,.01], 'hidden_layer_sizes': [(10,), (15,),(20,),(25,)]}
svc = MLPClassifier()
clf2 = GridSearchCV(svc, parameters, cv = 10, scoring = 'accuracy')
clf2.fit(player2, record)
print clf2.best_params_
print clf2.best_score_
print clf2.predict(player2test)
print ("Test accuracy: " + str(clf2.score(player2test, recordtest)))
y_score = clf2.predict_proba(player2test)
fpr, tpr, _ = roc_curve(recordtest, y_score[:,1])
roc_auc = auc(fpr, tpr)
afpr.append(fpr)
atpr.append(tpr)
aroc_auc.append(roc_auc)

parameters = {'max_iter': [25,50,75,100,125,150,200, 300], 'activation':['identity'], 'learning_rate_init': [.3,.2,.1,.05,.025,.01], 'hidden_layer_sizes': [(10,), (15,),(20,),(25,)]}
svc = MLPClassifier()
clf4 = GridSearchCV(svc, parameters, cv = 10, scoring = 'accuracy')
clf4.fit(team, record)
print clf4.best_params_
print clf4.best_score_
pred = clf4.predict(teamtest)
print ("Test accuracy: " + str(clf4.score(teamtest, recordtest)))
y_score = clf4.predict_proba(teamtest)
fpr, tpr, _ = roc_curve(recordtest, y_score[:,1])
roc_auc = auc(fpr, tpr)
afpr.append(fpr)
atpr.append(tpr)
aroc_auc.append(roc_auc)

Acc = metrics.accuracy_score(recordtest, pred)
Recall = metrics.recall_score(recordtest, pred)
Prec = metrics.precision_score(recordtest, pred)
print pred
print Acc
print Recall
print Prec

parameters = {'max_iter': [25,50,75,100,125,150,200, 300], 'activation':['identity'], 'learning_rate_init': [.3,.2,.1,.05,.025,.01], 'hidden_layer_sizes': [(10,), (15,),(20,),(25,)]}
svc = MLPClassifier()
clf3 = GridSearchCV(svc, parameters, cv = 10, scoring = 'accuracy')
clf3.fit(both, record)
print clf3.best_params_
print clf3.best_score_
pred = clf3.predict(bothtest)
print ("Test accuracy: " + str(clf3.score(bothtest, recordtest)))
y_score = clf3.predict_proba(bothtest)
fpr, tpr, _ = roc_curve(recordtest, y_score[:,1])
roc_auc = auc(fpr, tpr)
afpr.append(fpr)
atpr.append(tpr)
aroc_auc.append(roc_auc)


Acc = metrics.accuracy_score(recordtest, pred)
Recall = metrics.recall_score(recordtest, pred)
Prec = metrics.precision_score(recordtest, pred)
print pred
print Acc
print Recall
print Prec





names = ['player1','player2', 'both', 'team', '2player1','2player2', '2both', '2team']
# Plot all ROC curves
plt.figure(figsize=(20,10))

colors = ['aqua', 'orange', 'purple', 'green','red', 'black', 'pink', 'yellow']
for i, color in zip(range(8), colors):
    plt.plot(afpr[i], atpr[i], color=color, lw=2,
             label = names[i] + ' (area = {1:0.2f})'
             ''.format(i, aroc_auc[i]))


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate' )
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
