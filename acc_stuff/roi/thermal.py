"""
Thermal classifier 
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split, cross_val_score

DATA_FOLDER = "C:\\Users\\t-yeresh\\data"

data = pd.DataFrame.from_csv(os.path.join(DATA_FOLDER, "roi", "thermalClassifierData_1.csv"), index_col=4, header=None)

X = StandardScaler().fit_transform(data.values)
y = data.index

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.5)


clf = SVC(class_weight={1: 100, 2: 100}, probability=True)
y_score = clf.fit(x_train, y_train).decision_function(x_test)
fpr, tpr, thr = roc_curve(y_test.ravel(), y_score.ravel(), pos_label=2)


def plot():
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Now run the classifier on the unlabeled data

idx = np.where(fpr >= .05)[0][0]
theta = thr[idx]
print("Using theta=%f with %f FP and %f TP" % (theta, fpr[idx], tpr[idx]))

scaler = StandardScaler().fit(data.values)
newdata = pd.DataFrame.from_csv(os.path.join(DATA_FOLDER, "roi", "thermal_ToClassify.csv"), index_col=None, header=None)

clf = SVC(class_weight={1: 100, 2: 100}, probability=True).fit(scaler.transform(data.values), y)
out = clf.decision_function(scaler.fit_transform(newdata)) > theta

newdata['lbl'] = out.ravel().astype(int) + 1  # 1-> regular, 2->thermal
newdata.to_csv(os.path.join(DATA_FOLDER, "roi", "thermal_ToClassify__out.csv"), header=None, index=None,
               float_format="%.5f")











