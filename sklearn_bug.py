# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:25:09 2019

@author: arwen
"""


# In[] : bug 1 : scorer doesn't keep track of classes
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier

import numpy as np

X = np.random.randn(100,5)
y = np.array([0]*33 + [1]*33 + [2] * 33 + [4])

# test modif


estimator = DummyClassifier()

res = cross_validate(estimator,X,y,scoring="neg_log_loss") # => raise error
res = cross_validate(estimator,X,1*(y == 0),scoring="brier_score_loss")

res = cross_validate(estimator,X,y,scoring="roc_auc")

res = cross_validate(estimator,X,1*(y == 0),scoring="roc_auc")

from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss



#brier_score_loss_scorer
# In[] : bug 2
X = np.random.randn(100,5)

y = np.array([["a"] * 50 + ["b"]*50,["a"] * 50 + ["b"]*50]).T




from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


forest = RandomForestClassifier(n_estimators=10)
forest.fit(X,y)

yhat = forest.predict(X)
yhat_proba = forest.predict_proba(X)
forest.classes_


tree = DecisionTreeClassifier()
tree.fit(X,y)

yhat = tree.predict(X)
yhat_proba = tree.predict_proba(X)
tree.classes_


forest.fit(X,yd2[:,0])

yhat = forest.predict(X)
yhat_proba = forest.predict_proba(X)

