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
    
    
from sklearn.tree.tests.test_tree import CLF_TREES, REG_TREES,assert_equal,assert_array_equal, assert_almost_equal



def test_multioutput():
    # Check estimators on multi-output problems.
    X = [[-2, -1],
         [-1, -1],
         [-1, -2],
         [1, 1],
         [1, 2],
         [2, 1],
         [-2, 1],
         [-1, 1],
         [-1, 2],
         [2, -1],
         [1, -1],
         [1, -2]]

    y = [[-1, 0],
         [-1, 0],
         [-1, 0],
         [1, 1],
         [1, 1],
         [1, 1],
         [-1, 2],
         [-1, 2],
         [-1, 2],
         [1, 3],
         [1, 3],
         [1, 3]]

    T = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_true = [[-1, 0], [1, 1], [-1, 2], [1, 3]]

    # toy classification problem
    for name, TreeClassifier in CLF_TREES.items():
        clf = TreeClassifier(random_state=0)
        y_hat = clf.fit(X, y).predict(T)
        assert_array_equal(y_hat, y_true)
        assert_equal(y_hat.shape, (4, 2))
        
        
        proba = clf.predict_proba(T)
        assert_equal(len(proba), 2)
        assert_equal(proba[0].shape, (4, 2))
        assert_equal(proba[1].shape, (4, 4))

        with np.errstate(divide="ignore"): # rajout Lionel
            log_proba = clf.predict_log_proba(T)
            
        assert_equal(len(log_proba), 2)
        assert_equal(log_proba[0].shape, (4, 2))
        assert_equal(log_proba[1].shape, (4, 4))

    # toy regression problem
    for name, TreeRegressor in REG_TREES.items():
        reg = TreeRegressor(random_state=0)
        y_hat = reg.fit(X, y).predict(T)
        assert_almost_equal(y_hat, y_true)
        assert_equal(y_hat.shape, (4, 2))

# tree.tests.test_tree.py Line 935, a rajouter apres
def test_multioutput_with_object():
    # Check estimators on multi-output problems.
    X = [[-2, -1],
         [-1, -1],
         [-1, -2],
         [1, 1],
         [1, 2],
         [2, 1],
         [-2, 1],
         [-1, 1],
         [-1, 2],
         [2, -1],
         [1, -1],
         [1, -2]]

    y = [['-1', '0'],
         ['-1', '0'],
         ['-1', '0'],
         ['1', '1'],
         ['1', '1'],
         ['1', '1'],
         ['-1', '2'],
         ['-1', '2'],
         ['-1', '2'],
         ['1', '3'],
         ['1', '3'],
         ['1', '3']]
    
    y = np.array(y).astype(str)

    T = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_true = [['-1','0'], ['1', '1'], ['-1', '2'], ['1', '3']]

    # toy classification problem
    for name, TreeClassifier in CLF_TREES.items():
        clf = TreeClassifier(random_state=0)
        y_hat = clf.fit(X, y).predict(T)
        assert_array_equal(y_hat, y_true)
        assert_equal(y_hat.shape, (4, 2))

        proba = clf.predict_proba(T)
        assert_equal(len(proba), 2)
        assert_equal(proba[0].shape, (4, 2))
        assert_equal(proba[1].shape, (4, 4))

        log_proba = clf.predict_log_proba(T)
        assert_equal(len(log_proba), 2)
        assert_equal(log_proba[0].shape, (4, 2))
        assert_equal(log_proba[1].shape, (4, 4))
    

# In[]
import pytest
from sklearn.ensemble.tests.test_forest import FOREST_CLASSIFIERS,FOREST_ESTIMATORS,FOREST_CLASSIFIERS_REGRESSORS, assert_array_almost_equal
def check_multioutput(name):
    # Check estimators on multi-output problems.

    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [-2, 1],
               [-1, 1], [-1, 2], [2, -1], [1, -1], [1, -2]]
    y_train = [[-1, 0], [-1, 0], [-1, 0], [1, 1], [1, 1], [1, 1], [-1, 2],
               [-1, 2], [-1, 2], [1, 3], [1, 3], [1, 3]]
    X_test = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_test = [[-1, 0], [1, 1], [-1, 2], [1, 3]]

    est = FOREST_ESTIMATORS[name](random_state=0, bootstrap=False)
    y_pred = est.fit(X_train, y_train).predict(X_test)
    assert_array_almost_equal(y_pred, y_test)

    if name in FOREST_CLASSIFIERS:
        with np.errstate(divide="ignore"):
            proba = est.predict_proba(X_test)
            assert len(proba) == 2
            assert proba[0].shape == (4, 2)
            assert proba[1].shape == (4, 4)

            log_proba = est.predict_log_proba(X_test)
            assert len(log_proba) == 2
            assert log_proba[0].shape == (4, 2)
            assert log_proba[1].shape == (4, 4)


@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS_REGRESSORS)
def test_multioutput2(name):
    check_multioutput(name)



def check_multioutput_with_object(name):
    # Check estimators on multi-output problems.

    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [-2, 1],
               [-1, 1], [-1, 2], [2, -1], [1, -1], [1, -2]]
    y_train = [['-1', '0'], ['-1', '0'], ['-1', '0'], ['1', '1'], ['1', '1'], ['1', '1'], ['-1', '2'],
               ['-1', '2'], ['-1', '2'], ['1', '3'], ['1', '3'], ['1', '3']]
    X_test = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_test = [['-1', '0'], ['1', '1'], ['-1', '2'], ['1', '3']]

    est = FOREST_ESTIMATORS[name](random_state=0, bootstrap=False)
    y_pred = est.fit(X_train, y_train).predict(X_test)
    assert_array_almost_equal(y_pred, y_test)

    if name in FOREST_CLASSIFIERS:
        with np.errstate(divide="ignore"):
            proba = est.predict_proba(X_test)
            assert len(proba) == 2
            assert proba[0].shape == (4, 2)
            assert proba[1].shape == (4, 4)

            log_proba = est.predict_log_proba(X_test)
            assert len(log_proba) == 2
            assert log_proba[0].shape == (4, 2)
            assert log_proba[1].shape == (4, 4)


@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
def test_multioutput2_with_object(name):
    check_multioutput(name)




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

#print("toto")


#forest.py, predict 551
#            ### Fix 2 : ?? ###
#            all_predictions = []
#            for k in range(self.n_outputs_):
#                all_predictions.append(
#                        self.classes_[k].take(
#                                np.argmax(proba[:, k], axis=1),
#                                axis=0).reshape((-1,1)))
#            predictions = np.concatenate(all_predictions,axis=1)
            
            
#            n_samples = proba[0].shape[0]
#            predictions = np.zeros((n_samples, self.n_outputs_))
#
#            for k in range(self.n_outputs_):
#                predictions[:, k] = self.classes_[k].take(np.argmax(proba[k],
#                                                                    axis=1),
#                                                          axis=0)