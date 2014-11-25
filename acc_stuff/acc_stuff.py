from feature import spherical_feature
from factors import Factors
from feature_selection import forward_select
import numpy as np
import pandas as pd 



"""
Run the obs.csv file through the feature/factor maker and save to .csv file. 
"""
obs_path = "obs_partial.csv"
obs = pd.DataFrame.from_csv(obs_path, index_col=120, header= None)
obs_f = spherical_feature().compute(obs.values).astype(float) 
obs_f = pd.DataFrame(obs_f, index=obs.index)
obs_fact = Factors(labels=["AF", "PF", "WLK", "STD", "SIT"]).fit_transform(obs_f)

#pd.DataFrame(obs_f, index=obs.index).to_csv("f_obs_sphirical_plus_moments.csv", header=False, index=True)
#obs_fact.to_csv("f_obs_factors.csv", header=False, index=True)

"""
Featuer selection on the features 
"""
s = forward_select(obs_f, num_features=obs_f.shape[1], num_splits=5)
s.to_csv("forward_selection.csv")

