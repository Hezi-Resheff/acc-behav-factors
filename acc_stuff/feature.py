"""
Features of the ACC signal. 
Should have a base class, but...
"""

import numpy as np
import scipy.stats as stats 


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

if __name__ == "__main__":
    sample = np.zeros(120) + 2*np.random.rand(120) - 1
    samples= [sample for i in range(5)]

    f = simple_features().compute(samples)
    print(f)


