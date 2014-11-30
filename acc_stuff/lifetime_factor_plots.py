"""
How do the factor loadings during each behavior change over the lifetime? between animals? etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature import spherical_feature
from factors import Factors

data = pd.DataFrame.from_csv("C:\\Users\\heziresheff\\Documents\\data\\storks_acc_2012\\storks_2012_id_date_acc_behav_SAMPLE.csv", sep="\t", header=None)

# calc features
f = spherical_feature().compute(data[data.index==17940138][2].apply(lambda s: np.array(s.split(',')).astype(float)).values)

# factors obj -- fit 
factors = Factors(labels=["AF", "PF", "WLK", "STD", "SIT"]).fit(pd.DataFrame(data=f, index=data[data.index==17940138][3], columns=None))
print(factors._factors)
