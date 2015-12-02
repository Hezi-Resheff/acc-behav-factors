"""
Behav. clf. with probabilistic output. 
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

from calcstats import stat_calculator

PATH = "C:\\Users\\t-yeresh\\data\\Ron"
THETA = .5

def inspect():
    data = pd.DataFrame.from_csv(os.path.join(PATH, "obs.csv"), header=None, index_col=[0])
    features, labels = stat_calculator().get_stats_labels(data.values)

    f_train, f_test, l_train , l_test = train_test_split(features, labels, test_size=.5)

    clf = RandomForestClassifier(n_estimators=30).fit(f_train, l_train)

    out = clf.predict(f_test)

    out_p = clf.predict_proba(f_test)
    out_p = pd.DataFrame(out_p, columns=clf.classes_)

    # regular conf mat
    cm1 = confusion_matrix(l_test, out)

    # regular argmax confusion matrix 
    predicted = out_p.apply(lambda s: s.argmax(), axis=1)
    cm2 = confusion_matrix(l_test, predicted)

    # thresholded argmax confusion matrix 
    predicted2 = out_p.apply(lambda s: s.argmax() if s.max() > THETA else "zother", axis=1)
    cm2 = confusion_matrix(l_test, predicted2)

    print(cm1, cm2)


def train_test(train_f, test_f):
    data_train = pd.DataFrame.from_csv(os.path.join(PATH, train_f), header=None, index_col=[0])
    features_train, labels_train = stat_calculator().get_stats_labels(data_train.values)
    clf = RandomForestClassifier(n_estimators=30).fit(features_train, labels_train)
    classes = clf.classes_

    out = []
    for t_f in test_f:
        data_test_all = pd.DataFrame.from_csv(os.path.join(PATH, t_f))
        data_test = data_test_all["eobs:accelerations-raw"].apply(lambda r: pd.Series(r.split())).astype(float)
        features_test = stat_calculator().get_stats(data_test.values)
        
        prd_prob = clf.predict_proba(features_test)        
        prd = classes[prd_prob.argmax(axis=1)]
        pprd = prd_prob.max(axis=1)
        data_test_all["label"] = prd 
        data_test_all["label_prob"] = pprd
        out.append(data_test_all)

    return out 

if __name__ == "__main__":
    #inspect()

    files = ["MostlyFlight.csv", "FlightAndMore.csv", "FewFlights.csv"]
    out = train_test("obs.csv", files)
    for frame, in_file in zip(out, files):
        frame.to_csv(os.path.join(PATH, in_file.replace(".csv", "__out.csv")))