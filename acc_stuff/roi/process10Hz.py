"""
produce a tag_id, timestamp, behav file from the 10Hz data 
"""

import pandas as pd 
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix

from Calibrator import Calibrator
from calcstats import stat_calculator
from downsample4AcceleRater import downsample

def load_obs(obs_file):
    obs = []
    labels = []
    with open(obs_file, "r") as data:
        for row in data:            
            cells = row.split(',')
            vals, label = list(map(float, cells[:-1])), cells[-1].strip("\n")            
            nrows = len(vals) / 3                  
            # transform to X,Y,Z...
            vals = np.array(vals).reshape(3, nrows).T.ravel()
            obs.append(vals)
            labels.append(label)
    return obs, labels

def test_cross_val(obs_file):
    # load 
    data, labels = load_obs(obs_file)
    stats = stat_calculator().get_stats(data)

    #test
    data_train, data_test, l_train, l_test =  train_test_split(stats, labels, train_size=.5)  
    out_l = RandomForestClassifier().fit(data_train, l_train).predict(data_test)
    cm = confusion_matrix(l_test, out_l)
    cm = pd.DataFrame(cm, index=np.unique(labels), columns=np.unique(labels))
    print(cm)
       

def process10Zdata(source_file, obs_file, calib_file):
    # load
    data = pd.DataFrame.from_csv(source_file, index_col='tag_id')      
    acc_data = data.apply(lambda row: pd.Series(np.array(row.acc_data.split(",")).astype(float)) , axis=1)    
    
    # calib, downsample
    data_calibrated = Calibrator(calib_file).transfom_all(acc_data)
    data_downsampled = downsample(data_calibrated)
    
    # behavs
    data_obs, labels_train = load_obs(obs_file)
    stats_train = stat_calculator().get_stats(data_obs)    
    stats_test = stat_calculator().get_stats(data_downsampled.values)
    lbls = RandomForestClassifier().fit(stats_train, labels_train).predict(stats_test)
    
    # pub together and save 
    data["behav"] = lbls
    print(data[["date", "time", "behav"]])



if __name__ == "__main__":
    root_dir = "c:\\data\\Vultures"
    obs_file = os.path.join(root_dir, "vultures.csv")
    calib_file = os.path.join(root_dir, "params.csv")
    data_file_10hz = os.path.join(root_dir, "Vultures-10hz-10K-rand.csv")
    #test_cross_val(obs_file)
    process10Zdata(data_file_10hz, obs_file, calib_file)