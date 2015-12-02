"""
Two stage scheme:
1) Is it in the known set?
2) What is it?
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from calcstats import stat_calculator

NU = .005
GAMMA = .01

# TODO: make this an sklearn predictor object 
class TwoStage(object):

    def __init__(self, *args, **kwargs):
        super(TwoStage, self).__init__(*args, **kwargs)
        self._oneCls = OneClassSVM(nu=NU, gamma=GAMMA)
        self._clf = RandomForestClassifier(n_estimators=30)
        self._scaler = StandardScaler()

    def fit(self, data, labels):
        sdata = self._scaler.fit_transform(data)
        self._oneCls.fit(sdata)
        self._clf.fit(sdata, labels)
        return self

    def predict(self, data):
        sdata = self._scaler.transform(data)
        is_known_cls = self._oneCls.predict(sdata)
        cls = self._clf.predict(sdata)
        cls[is_known_cls == -1] = "zother"        
        classes = list(self._clf.classes_) + ["zother"]
        return cls, classes


def test():
    data = pd.DataFrame.from_csv(os.path.join(PATH, "obs.csv"), header=None, index_col=[0])
    features, labels = stat_calculator(what_to_calc=list(range(17))).get_stats_labels(data.values)
    f_train, f_test, l_train , l_test = train_test_split(features, labels, test_size=.5)
    
    out, classes = TwoStage().fit(f_train, l_train).predict(f_test)
    cm = confusion_matrix(l_test, out)    
    cm = pd.DataFrame(cm, columns=classes, index=classes)
    print(cm)
    cm.to_csv(os.path.join(PATH, "obs_cm__out.csv"))


def label_files(train_f, file_list):

    # Train on all the training data 
    print("Training...")
    data_train = pd.DataFrame.from_csv(os.path.join(PATH, train_f), header=None, index_col=[0])
    features_train, labels_train = stat_calculator(what_to_calc=list(range(17))).get_stats_labels(data_train.values)
    clf = TwoStage().fit(features_train, labels_train)

    # For each of the files in `file_list` -- compute and save output
    for t_f in file_list:
        print("Processing file: {}".format(t_f))
        data_test_all = pd.DataFrame.from_csv(os.path.join(PATH, t_f))
        data_test = data_test_all["eobs:accelerations-raw"].apply(lambda r: pd.Series(r.split())).astype(float)
        features_test = stat_calculator(what_to_calc=list(range(17))).get_stats(data_test.values)
        data_test_all["label"] = clf.predict(features_test)[0]
        data_test_all.to_csv(os.path.join(PATH, t_f.replace(".csv", "__out.csv")))

PATH = "C:\\Users\\t-yeresh\\data\\Ron"

if __name__ == "__main__":
    print("Starting...")
    test()
    label_files("obs.csv", ["MostlyFlight.csv", "FlightAndMore.csv", "FewFlights.csv"])
    print("Done!")