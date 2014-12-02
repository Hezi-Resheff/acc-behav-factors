"""
How do the factor loadings during each behavior change over the lifetime? between animals? etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature import spherical_feature, simple_features
from factors import Factors

print("reading data")
data = pd.DataFrame.from_csv("C:\\Users\\heziresheff\\Documents\\data\\storks_acc_2012\\storks_2012_id_date_acc_behav.csv", sep="\t", header=None)
print("reducing data")
data = data.iloc[np.random.randint(0, data.shape[0], 1000000)]
print("computing acc series")
acc_series = data[2].apply(lambda s: np.array(s.split(',')).astype(float))
print("computing acc features")
acc_features = simple_features().compute(acc_series)
del acc_series

# overall factors (all animals)
print("computing factors")
facts = Factors(labels=["AF", "PF", "WLK", "STD", "SIT"]).fit(pd.DataFrame(data=acc_features, index=data[3], columns=None))


unique_animals = np.unique(data.index)
for animal in unique_animals:
    print("bird id: ", animal)
    this_flight_samples = ((data.index == animal) & (data[3] == 2)).values
    this_factors = facts.transform(pd.DataFrame(acc_features[this_flight_samples], index=data[this_flight_samples][1])).sort()
    print("# samples: ", len(this_factors))
    this_factors.plot(style="-x", title="animal: %i"%animal)
    plt.show()

