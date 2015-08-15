"""
copy of DSAA/featuers 
"""

import numpy as np
import pandas as pd 
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

class FeatureBase(object):
    def __init__(self, normalize_atches = False): pass
    def compute(self, samples):
        return np.array([self._compute(sample) for sample in samples])


class MultiScalePatches(FeatureBase):
    """
    Represent an ACC sample with a patch-codebook on multiple scales 
    """
    def __init__(self, scales, size, normalize_patches=False):
        """
        scales - list of scaels to use 
        size - list of sized of codebooks for each scals | int for 1 size 4 all
        normalize_patches - bool, if True then each patch is normalized to have mean 0 and std 1
        """
        self._scales = scales
        self._size = np.ones_like(scales)*size
        self._norm_patches = normalize_patches
        self._models = [{'scale':scale, 'cbsize':sz, 'patches':[], 'km':MiniBatchKMeans(n_clusters=sz)} \
                        for scale, sz in zip(self._scales, self._size)]

    def _get_patches(self, row, scale):
        data = np.array([row[:, i:i+scale].ravel() for i in range(row.shape[1]-scale)])
        if self._norm_patches:
            data = MinMaxScaler().fit_transform(data.T).T
        return data

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
        out = np.hstack(rep)
        return out.astype(float)/out.sum()


