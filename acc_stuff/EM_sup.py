"""
Model the simple_feature representation as a mixture of Gaussians per class. 
"""
from local_settings import DATA_FOLDER
from feature import simple_features

from sklearn.mixture import GMM
import pandas as pd
import numpy as np
import os


class GMM_clf(object):
    def __init__(self, nCompCls=2):
        self._ncpc = nCompCls
        self._labels = []
        self._gmms = []

    def fit(self, X, y): 
        self._labels = np.unique(y)


    def predict(self, X): pass 

data = pd.DataFrame.from_csv(os.path.join(DATA_FOLDER, "storks2012", "obs.csv"),index_col=120, header= None)
data = simple_features().compute(data.values)

print(data)