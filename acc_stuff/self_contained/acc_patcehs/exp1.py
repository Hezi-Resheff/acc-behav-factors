"""
Part of the "Stage B" research proposal. 
Classify ACC signals with patches features. Compare to raw data & AcceleRater features. 
"""
import os 
import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score

from calcstats import stat_calculator


class FeatureBase(object):
    def __init__(self): pass
    def compute(self, samples):
        return np.array([self._compute(sample) for sample in samples])

class simple_features(FeatureBase):
    def _compute(self, sample): 
        x, y, z = np.reshape(sample, (len(sample)/3,3)).T 
        m = [[f(x), f(y), f(z)] for f in (np.mean, np.std, stats.skew, stats.kurtosis) ]
        return np.array(m).flatten() 

class AcceleRaterFeatures(FeatureBase):
    def _compute(self, sample):
        return np.array(stat_calculator()._calc_row(sample))


class MultiScalePatches(FeatureBase):
    """
    Represent an ACC sample with a patch-codebook on multiple scales 
    """
    def __init__(self, scales, size):
        """
        scales - list of scaels to use 
        size - list of sized of codebooks for each scals | int for 1 size 4 all 
        """
        self._scales = scales
        self._size = np.ones_like(scales)*size 
        self._models = [{'scale':scale, 'cbsize':sz, 'patches':[], 'km':MiniBatchKMeans(n_clusters=sz)} \
                        for scale, sz in zip(self._scales, self._size)]


    def _get_patches(self, row, scale):
        return [row[:, i:i+scale].ravel() for i in range(row.shape[1]-scale)]

    def _add_patches(self, row):
        for model in self._models:            
            model['patches'].extend(self._get_patches(row, model['scale']))
        
    def fit(self, data):
        for row in data:
            l = len(row)/3
            self._add_patches(row.reshape(l, 3).T)
        for model in self._models:
            model['km'].fit(model['patches'])
            model['patches'] = []
        return self

    def _compute(self, row):
        l = len(row)/3
        row = row.reshape(l, 3).T
        rep = [np.histogram(model['km'].predict(self._get_patches(row, model['scale'])), np.arange(model['cbsize']+1))[0] for model in self._models]
        return np.hstack(rep)



def go():
    # this is the pipeline to use
    pipeline = Pipeline([('scaler', StandardScaler()), 
                         ('classifier', SVC())
                       ])

    
    data_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "storks_obs_train.csv")
    data = [row for row in open(data_file, "r")]
    
    X = np.array([row[:-1].split(',') for row in data])
    y = [row[-1] for row in data]
    print(X, y)

    # TODO: strip the /n for the row in data, make float, continue... 

if __name__ == "__main__":
    go()

