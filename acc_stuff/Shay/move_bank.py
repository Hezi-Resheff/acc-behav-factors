"""
Process the movebank output files 
"""

import numpy as np
import pandas as pd 
import os 


def process_acc_file(f_name):
    # load the rew data 
    f = pd.DataFrame.from_csv(f_name) 

    # keep only these columns 
    mapping = {'eobs:accelerations-raw':'acc_data', 
               'tag-local-identifier':'tag_id', 
               'individual-local-identifier':'bird_id'}
    cols_keep = mapping.keys()
    out = f[cols_keep].copy()
    out.columns = [mapping[key] for key in out.columns]

    # process 
    out['time'] = pd.DatetimeIndex(f['eobs:start-timestamp']).time
    out['date'] = pd.DatetimeIndex(f['eobs:start-timestamp']).date
    out['bird_id'] = out['bird_id'].apply(lambda s: s[:4] + '0' + s[-4:-1])
    out['num_samples'] = out['acc_data'].apply(lambda s: s.count(' ') + 1)

    return out

def process_gps_file(f_name):
    pass

def insert(frame, table):
    pass 

test_file = "C:\\Users\\heziresheff\\Downloads\\acc.csv"
out = process_acc_file(test_file)
print(out)
