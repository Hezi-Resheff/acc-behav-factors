"""
Model the simple_feature representation as a mixture of Gaussians per class. 
"""
from local_settings import DATA_FOLDER
from feature import simple_features

from sklearn.cross_validation import StratifiedKFold
from sklearn.mixture import GMM
import pandas as pd
import numpy as np
import os


class GMM_clf():
    def __init__(self, nCompCls=1):
        self._ncpc = nCompCls
        self._labels = []
        self._gmms = []

    def fit(self, X, y): 
        self._labels = np.unique(y)
        self._gmms = []
        for lbl in self._labels:
            _X = X[y == lbl]
            _gmm = GMM(n_components=self._ncpc, covariance_type="full").fit(_X)
            self._gmms.append(_gmm)
        return self

    def predict(self, X): 
        out = [gmm.score(X) for gmm in self._gmms] 
        return self._labels[np.argmax(out, 0)]

    def score(self, X, y, k):
        _scores = []
        for train, test in StratifiedKFold(y, n_folds=k):
            _scores.append((self.fit(X[train], y[train]).predict(X[test]) == y[test]).mean())
        return _scores

data = pd.DataFrame.from_csv(os.path.join(DATA_FOLDER, "storks2012", "obs.csv"),index_col=120, header= None)
X = simple_features().compute(data.values)
print(GMM_clf().score(X, data.index.values, 3))
