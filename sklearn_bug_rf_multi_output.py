# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:25:09 2019

@author: arwen
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np

def test_RandomForestClassifier_multioutput_object():
    X = np.random.randn(100,5)
    y = np.array([["a"] * 50 + ["b"]*50,["a"] * 50 + ["b"]*50]).T
    
    forest = RandomForestClassifier(n_estimators=10, random_state=123)
    
    forest.fit(X,y)
    
    yhat = forest.predict(X)
    assert yhat.shape == (100,2)

def test_DecisionTree_multioutput_object():
    X = np.random.randn(100,5)
    y = np.array([["a"] * 50 + ["b"]*50,["a"] * 50 + ["b"]*50]).T
    
    forest = DecisionTreeClassifier(random_state=123)
    
    forest.fit(X,y)
    
    yhat = forest.predict(X)
    assert yhat.shape == (100,2)
# In[]
#tree.py# predict Line 427
#
### Fix 1 : ??
#                predictions = np.zeros((n_samples, self.n_outputs_)) # Fix Lionel, dtype=self.classes_[0].dtype)
#
#                for k in range(self.n_outputs_):
#                    predictions[:, k] = self.classes_[k].take(
#                        np.argmax(proba[:, k], axis=1),
#                        axis=0)

### Fix 2 : ?? ###
#all_predictions = []
#for k in range(self.n_outputs_):
#    all_predictions.append(
#        self.classes_[k].take(
#            np.argmax(proba[:, k], axis=1),
#            axis=0).reshape((-1, 1)))
#predictions = np.concatenate(all_predictions, axis=1)

print("toto")
