"""
produce a tag_id, timestamp, behav file from the 10Hz, 3.33Hz data 
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


class Learner(RandomForestClassifier):    
    @staticmethod
    def _load(obs_file):
        data, labels = load_obs(obs_file)
        stats = stat_calculator().get_stats(data)
        return stats, labels
   
    def fit_w_obs(self, obs_file):                
        return self.fit(*Learner._load(obs_file))
       
    def cross_validate(self, obs):
        data_train, data_test, l_train, l_test =  train_test_split(*Learner._load(obs_file), train_size=.5)  
        out_l = self.fit(data_train, l_train).predict(data_test)
        return pd.DataFrame(confusion_matrix(l_test, out_l), index=np.unique(l_train), columns=np.unique(l_train))
        
   

def process10Hzdata(source_file, calib_file):
    # load
    data = pd.DataFrame.from_csv(source_file, index_col='tag_id').iloc[:10]      
    acc_data = data.apply(lambda row: pd.Series(np.array(row.acc_data.split(",")).astype(float)) , axis=1)    
    
    # calib, downsample
    data_calibrated = Calibrator(calib_file).transfom_all(acc_data)
    data_downsampled = downsample(data_calibrated)
    
    #compute stats
    stats= stat_calculator().get_stats(data_downsampled.values)
   
    return data[["date","time","num_samples"]], stats
   
        
def process3Hzdata(source_file, obs_file, calib_file):
    # load 
    # calib   
    pass

def driver(calib, obs, data10, data3):
    # Train
    clf = Learner().fit_w_obs(obs)

    # 10Hz data
    data10, stats10 = process10Hzdata(data_file_10hz , calib_file)
    lbl10 = clf.predict(stats10)
    data10["label"] = lbl10
    print(data10)

    """
    data3, stats3 = None, None
    lbl3 = clf.predict(stats3)
    data3["label"] = lbl
    print(data3)
    """

if __name__ == "__main__":
    root_dir = "c:\\data\\Vultures"
    obs_file = os.path.join(root_dir, "vultures.csv")
    calib_file = os.path.join(root_dir, "params.csv")
    data_file_10hz = os.path.join(root_dir, "Vultures-10hz-10K-rand.csv")
    data_file_3Hz = None
    driver(calib_file, obs_file ,data_file_10hz, data_file_3Hz)
    print(Learner().cross_validate(obs_file))
   
    