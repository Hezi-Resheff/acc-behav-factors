"""
Factor loading over the year with the semi-supervised factors. 
"""

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from local_settings import DATA_FOLDER, PLOTS_OUT
from feature import simple_features
from factors import FactorsSemiSup
from plots import plot_factors

DAILY_AGGREGATION = True

# load ACC data
print("Loading ACC")
data = pd.DataFrame.from_csv(os.path.join(DATA_FOLDER,"storks2012","storks_2012_id_date_acc_behav_downsampledX30.csv"), sep="\t", header=None)
acc_data = data[2].apply(lambda s: np.array(s.split(',')).astype(float)).values

# load GPS data 
print("Loading GPS")
gps_data = pd.DataFrame.from_csv(os.path.join(DATA_FOLDER,"Storks2012","storks_2012_GPS_sparse.csv"), header=None)
gps_data.columns = ['date', 'time', 'lat', 'long'] 


unique_animals = np.unique(data.index)

for animal_id in unique_animals:
    try: 
        print("Processing animal ID: ", animal_id)

        # calc features
        feature_frame = pd.DataFrame(data=simple_features().compute(acc_data[data.index==animal_id]),
                                        index=data[data.index==animal_id][3], #behav-code 
                                        columns=None)

        # transform to factors 
        labels = [["AF", "PF", "WLK", "STD", "SIT"][i-2] for i in np.unique(feature_frame.index)]
        factor_loading = FactorsSemiSup(labels=labels).fit_transform(feature_frame)
        factor_loading.index = pd.DatetimeIndex(data[data.index==animal_id][1]) #date 
        
        if DAILY_AGGREGATION:
            factor_loading = factor_loading.groupby(factor_loading.index).mean()[['AF', 'PF']]
            behav = None # if averaging then can't show the actualy behav... 
        else:
            #true behav
            behav = pd.DataFrame(data=np.array([feature_frame.index==behav for behav in np.unique(feature_frame.index)]).T, 
                            index=factor_loading.index, 
                            columns=labels)

        gps = pd.DataFrame(data=gps_data[gps_data.index == animal_id])
        gps.index = pd.DatetimeIndex(gps.date) 
        gps = gps[['lat','long']]
        gps = gps.groupby(gps.index).mean() #daily mean coordinates 

        plot_factors(factor_loading, behav, gps, where_behav=None, show_true_behav=not DAILY_AGGREGATION)

        plt.gcf().set_size_inches(18.5, 10.5)
        plt.savefig(os.path.join(PLOTS_OUT, "semi-sup-factors","factors_daily_agg", str(animal_id) + ".png"), bbox_inches='tight', dpi=300)
        #plt.show()

    except:
        print("Failed to plot for animal: ", animal_id)
    