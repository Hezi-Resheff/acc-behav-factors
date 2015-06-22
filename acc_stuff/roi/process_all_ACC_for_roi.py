"""
produce a tag_id, timestamp, behav file from the 10Hz, 3.33Hz data 
"""

import pandas as pd 
import numpy as np
import os
import pickle
import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from Calibrator import Calibrator
from calcstats import stat_calculator, statNames
from odba import odba

def downsample(frame):
    """Downsample from 10->3.3 (120 dp) by averaging every 3 points discarding the 40th on each axis. format is X,Y,Z,X,Y,Z..."""
    def downsample_row(row):        
        X, Y, Z = row[0::3], row[1::3], row[2::3]
        X = X[:-1].reshape(13, 3).T[0]
        Y = Y[:-1].reshape(13, 3).T[0]
        Z = Z[:-1].reshape(13, 3).T[0]
        return pd.Series(np.vstack([X, Y, Z]).T.ravel())
    return frame.apply(downsample_row, axis=1)

def load_obs(obs_file, print_odba=False):
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
    if print_odba:
        print("ODBA per behavior (mean):")   
        print(pd.DataFrame([odba(s) for s in obs], labels).groupby(labels).mean())
    return obs, labels


class Learner(RandomForestClassifier):        
 
    @staticmethod
    def _load(obs_file):
        data, labels = load_obs(obs_file)
        stats = stat_calculator().get_stats(data)
        return stats, labels
   
    def fit_w_obs(self, obs_file):                
        X, y = Learner._load(obs_file)            
        return self.fit(X, y)
       
    def cross_validate(self, obs_file):
        data_train, data_test, l_train, l_test =  train_test_split(*Learner._load(obs_file), train_size=.5)  
        out_l = self.fit(data_train, l_train).predict(data_test)
        return pd.DataFrame(confusion_matrix(l_test, out_l), index=np.unique(l_train), columns=np.unique(l_train))
           

def process_input_data(source_file, calib_file, sample3Hz=True):
    # load
    data = pd.DataFrame.from_csv(source_file)
       
    # acc    
    acc_data = data.apply(lambda row: pd.Series(np.array(row.acc_data.split(",")).astype(float)) , axis=1)    
            
    # calib
    processed_acc = Calibrator(calib_file).transfom_all(acc_data)       

    #  downsample
    if sample3Hz:
        processed_acc = downsample(processed_acc)

   # compute ODBA
    data["odba"] = processed_acc.apply(odba, axis=1)
    
    #compute stats
    stats= stat_calculator().get_stats(processed_acc.values)   
   
    return data[["date","time","num_samples", "odba"]], stats
   

def driver(root_dir, calib, obs, dir_in, dir_out, sample=True, clf_path=None):
       
    if clf_path is not None:
        print("Loading classifier...")
        clf =  joblib.load(clf_path)
    else:
         # Train and save 
        cm = Learner().cross_validate(obs)
        print(cm)
        clf = Learner().fit_w_obs(obs)                
        joblib.dump(clf, os.path.join(root_dir, "clf", "clf_vultures")) 
    
    # First the 10Hz data
    file_list = os.listdir(dir_in)
    for f in file_list:
        data, stats = process_input_data(os.path.join(dir_in, f), calib, sample3Hz=sample)
        data["label"] = clf.predict(stats)
        data.to_csv(os.path.join(dir_out, f))

   
def split(path, out_dir):
    """Split by tag_id """
    # load
    data = pd.DataFrame.from_csv(path, index_col=0, header=None)
    data.columns = ["date", "time", "num_samples", "acc_data"]

    # clean
    drop_list = [178, 179, 496, 629, 1559, 2305, 2553, 2569, 2571, 2584, 2746, 2749, 3059, 3083, 3724, 3725]
    data.drop(data.index[data.index.isin(drop_list)], inplace=True) # these are not Vultures

    for name, frame in data.groupby(data.index):
        frame.to_csv(os.path.join(out_dir, str(name)+".csv"))

        
if __name__ == "__main__":
    root_dir = "c:\\data\\Vultures"   
    clf_path = os.path.join(root_dir, "clf", "clf_vultures")
    calib = os.path.join(root_dir, "params.csv")
    obs = os.path.join(root_dir, "vultures.csv")
    
    data10_dir = os.path.join(root_dir,"tags", "10")
    data3_dir = os.path.join(root_dir, "tags", "3")
    data10_dir_out = os.path.join(root_dir, "out", "10")
    data3_dir_out = os.path.join(root_dir, "out", "3")
    
    """
    tag_dir_10 = os.path.join(root_dir, "tags", "10")
    tag_dir_3 = os.path.join(root_dir, "tags", "3")
    split(os.path.join(root_dir, "vultures3hz_all.csv"), tag_dir_3)
    """


    #driver(root_dir, calib, obs, data3_dir, data3_dir_out,sample=False, clf_path=clf_path)
    driver(root_dir, calib, obs, data10_dir, data10_dir_out,sample=True, clf_path=clf_path)
    
    
   
    