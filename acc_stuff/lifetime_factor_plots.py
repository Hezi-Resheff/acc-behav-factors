"""
How do the factor loadings during each behavior change over the lifetime? between animals? etc.
"""

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from local_settings import DATA_FOLDER
from feature import simple_features
from factors import Factors

data = pd.DataFrame.from_csv(os.path.join(DATA_FOLDER,"storks2012","storks_2012_id_date_acc_behav_SAMPLE.csv"), sep="\t", header=None)
acc_data = data[2].apply(lambda s: np.array(s.split(',')).astype(float)).values


unique_animals = np.unique(data.index)

for animal_id in unique_animals:
    # calc features
    feature_frame = pd.DataFrame(data=simple_features().compute(acc_data[data.index==animal_id]),
                                 index=data[data.index==animal_id][3], #behav-code 
                                 columns=None)

    # transform to factors 
    labels = [["AF", "PF", "WLK", "STD", "SIT"][i-2] for i in np.unique(feature_frame.index)]
    factor_loading = Factors(labels=labels).fit_transform(feature_frame)
    factor_loading.index = data[data.index==animal_id][1] #date 

    #plot factor loading 
    factor_loading.plot(style="-x")

    #plot true behav
    pd.DataFrame(data=np.array([feature_frame.index==behav for behav in np.unique(feature_frame.index)]).T, 
                 index=factor_loading.index, 
                 columns=["AF", "PF", "WLK", "STD", "SIT"]).plot(style="-x")

    #TODO: plot GPS 

    plt.show()

    #TODO: merge the two graphs above!