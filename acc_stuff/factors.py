"""
Try the factorization stuff with non-negative least squares

\alpha = argmin || s - F*\alpha||
s.t. \alpha >= 0

This makes s a conical combination of the factors (F columns)
"""

import numpy as np
import pandas as pd 
import scipy.stats as stats
from scipy.optimize import nnls


class Factors(object):

    def __init__(self, labels):  
        self._labels = labels

    def fit(self, samples): 
        """Fit the factors model
        samples: pd.DataFrame with the label as the index .
        """
        self._factors = np.array(samples.groupby(samples.index).mean())
        return self

    def transform(self, samples):
        """Transform to the factor space 
        """ 
        ndata = np.array([nnls(self._factors.T, s)[0] for s in samples.values])
        ndata = pd.DataFrame(ndata, samples.index, columns=self._labels)
        return ndata 

    def fit_transform(self, samples): 
        return self.fit(samples).transform(samples)