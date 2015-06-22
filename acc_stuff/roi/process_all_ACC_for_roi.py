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
    data = pd.DataFrame.from_csv(source_file, index_col=0, header=None)[:10000]
    data.drop(data.index[data.index.isin([1559, 2749, 3083])], inplace=True) # these are not Vultures
    data.columns = ["date", "time", "num_samples", "acc_data"]
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
   

def driver(root_dir, calib, obs, data10, data3, clf_path=None):
    # obs, calib
    obs_file = os.path.join(root_dir, obs)
    calib_file = os.path.join(root_dir, calib)
    # in
    data_file_10hz = os.path.join(root_dir, data10)
    data_file_3Hz = os.path.join(root_dir, data3)
    # out
    out_10 = os.path.join(root_dir, "vultures10hz__out.csv")
    out_3 = os.path.join(root_dir,  "vultures3hz__out.csv")

    if clf_path is not None:
        print("Loading classifier...")
        clf =  joblib.load(clf_path)
    else:
         # Train and save 
        cm = Learner().cross_validate(obs_file)
        print(cm)
        clf = Learner().fit_w_obs(obs_file)                
        joblib.dump(clf, os.path.join(root_dir, "clf", "clf_vultures")) 
    
    # 10Hz data
    data10, stats10 = process_input_data(data_file_10hz , calib_file, sample3Hz=True)
    lbl10 = clf.predict(stats10)
    data10["label"] = lbl10
    print(data10["odba"].groupby(data10.label).mean())
    data10.to_csv(out_10)
    

    data3, stats3 = process_input_data(data_file_3Hz , calib_file, sample3Hz=False)
    lbl3 = clf.predict(stats3)
    data3["label"] = lbl3
    print(data3["odba"].groupby(data3.label).mean())
    data3.to_csv(out_3)
    
if __name__ == "__main__":
    root_dir = "c:\\data\\Vultures"   
    clf_path = os.path.join(root_dir, "clf", "clf_vultures")
    driver(root_dir, "params.csv", "vultures.csv" ,"vultures10hz_all.csv", "vultures3hz_all.csv", None)
    
   
    