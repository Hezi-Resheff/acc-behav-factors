"""

1) Try the factorization stuff with non-negative least squares

\alpha = argmin || s - F*\alpha||
s.t. \alpha >= 0

This makes s a conical combination of the factors (F columns)

2) Same idea but with matrix factorization 
   samples(S) = factors(F)*loadings(A)
   
   F,A = argmin ||S-FA||
   s.t. A>=0
   
   can solve this with alternating ls/nnls 
   
   comment: unlike in (1) this should be done for the entire dataset and not for each 
   entity on it's own, or the factors will not be inerpretable.  
"""

import numpy as np
import pandas as pd 
import scipy.stats as stats
from scipy.optimize import nnls


class FactorsSemiSup(object):

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


class FactorsUnSup(object):
    def __init__(self, k):
        """
        k - the number of factors to produce 
        """
        self._k = k

    def fit(samples):
        self._fit(samples)
        return self

    def transform(samples): pass
    
    def fit_transform(samples): 
        return self._fit(samples, return_a=True)

    def _fit(samples, return_a = False):
        pass 
