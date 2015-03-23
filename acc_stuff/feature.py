"""
Features of the ACC signal. 
Should have a base class, but...
"""

import numpy as np
import pandas as pd 
import os 
import scipy.stats as stats 
from sklearn.cluster import MiniBatchKMeans

from local_settings import *

"""
features are the histogram of transitions between +/-/0 values of ACC in each axis
"""
class transition_feature(object):

    def __init__(self, threshold):
        self._threshold = threshold

    def compute(self, samples):
        return np.array([self._compute(sample) for sample in samples])

    def _compute(self, sample):
        sample = sample.reshape(len(sample)/3, 3).T
        
        sample_x = sample[0]
        f_x = self._compute_transition_histogram(sample_x)

        sample_y = sample[1]
        f_y = self._compute_transition_histogram(sample_y)
        
        sample_z = sample[2]
        f_z = self._compute_transition_histogram(sample_z)

        return np.hstack((f_x, f_y, f_z))

    def _compute_transition_histogram(self, vector):
       """  Copute the histogram of transition in the vector, between states +/-/0
           Return a flat vector to be used as a feature vector.  
           states are:
            - 1: above threshold
            - 2: below -threshold
            - 0: 0.w.
       """
       mat = np.zeros((3, 3)).astype(int)
       q_vector = np.zeros_like(vector)
       q_vector[vector > self._threshold] = 1
       q_vector[vector < -self._threshold] = 2
       
       for i in range(len(vector) - 1):
           mat[q_vector[i]][q_vector[i+1]] += 1

       return mat.flatten()

class spherical_feature(object):
    def __init__(self): pass 
    
    def compute(self, samples):
        return np.array([self._compute(sample) for sample in samples])

    def _compute(self, sample):
       
        # the sphirical part 
        x, y, z = np.reshape(sample, (len(sample)/3,3)).T 
        r = np.sqrt(x**2 + y**2 + z**2) 
        theta = np.arccos(z/r)  #[0, pi]
        thi = np.arctan2(y, x)  #[-pi, pi]
        
        # descretize
        r = (r/8).astype(int); r[r>2] = 2 
        theta = (theta / (np.pi/4)).astype(int) % 4    
        thi =   ((thi+np.pi) /(np.pi/2)).astype(int) % 4

        # hist 
        f_mat = np.zeros((3, 4, 4)) #3*4*4 = 48 options      
        for i in range(len(r)):
            f_mat[r[i], theta[i], thi[i]] += 1
        
        # now compute the moment part 
        moments = [[f(x), f(y), f(z)] for f in (np.mean, np.std, stats.skew, stats.kurtosis) ]


        return np.hstack((np.array(moments).flatten(), f_mat.flatten()))


class simple_features(object):
    def __inin__(self): pass 
    
    def compute(self, samples):
        return np.array([self._compute(sample) for sample in samples])

    def _compute(self, sample): 
        x, y, z = np.reshape(sample, (len(sample)/3,3)).T 
        m = [[f(x), f(y), f(z)] for f in (np.mean, np.std, stats.skew, stats.kurtosis) ]
        return np.array(m).flatten() 


class MultiScalePatches(object):
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
            self._add_patches(row.reshape(40, 3).T)
        for model in self._models:
            model['km'].fit(model['patches'])
            model['patches'] = []
        return self

    def _transform(self, row):
        rep = [np.histogram(model['km'].predict(self._get_patches(row, model['scale'])), np.arange(model['cbsize']+1))[0] for model in self._models]
        return np.hstack(rep)

    def transform(self, data):
        return np.array([self._transform(row.reshape(40, 3).T) for row in data])


if __name__ == "__main__":
    data = pd.DataFrame.from_csv(os.path.join(DATA_FOLDER, "storks2012", "obs.csv"),index_col=120, header= None)
    data.index.name = ''
    ftr = MultiScalePatches(scales=[3, 5, 7], size=20).fit(data.values)
    out = ftr.transform(data.values)
    pd.DataFrame(out, index=data.index).to_csv(os.path.join(DATA_FOLDER, "storks2012", "obs_multiscale_codebook35.csv"))
