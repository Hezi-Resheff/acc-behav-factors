from feature import spherical_feature
from factors import Factors
import numpy as np
import pandas as pd 

obs_path = "obs.csv"


obs = pd.DataFrame.from_csv(obs_path, index_col=120, header= None)
obs_f = spherical_feature().compute(obs.values).astype(float) 
obs_f = pd.DataFrame(obs_f, index=obs.index)
obs_fact = Factors(labels=["AF", "PF", "WLK", "STD", "SIT"]).fit_transform(obs_f)
#pd.DataFrame(obs_f, index=obs.index).to_csv("f_obs_sphirical_plus_moments.csv", header=False, index=True)
obs_fact.to_csv("f_obs_factors.csv", header=False, index=True)

