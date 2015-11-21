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

PATH = ""
THETA = .75

data = pd.DataFrame.from_csv(os.path.join(PATH, "obs.csv"), header=None, index_col=[0])
features, labels = stat_calculator().get_stats_labels(data.values)

f_train, f_test, l_train , l_test = train_test_split(features, labels, test_size=.5)

clf = RandomForestClassifier().fit(f_train, l_train)

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
