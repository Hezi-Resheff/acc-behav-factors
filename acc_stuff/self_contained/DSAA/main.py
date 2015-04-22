"""
This is the driver for all the experiments. 
Running this should reproduce all the reported results. 
"""

import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import os

from features import *
from util import *

DATA_PATH = ""


def load(data_file, sample_size=3000):
    """
    Load the raw data
    assume format is(\t sep columns):
    id    date    acc_data    label
    and acc_data format is:
    valX1,valY1,valZ1,valX2... 
    -
    Then sample sample_size from each class 
    """
    # Load data 
    data_frame = pd.DataFrame.from_csv(data_file, header=None, sep="\t")
    acc_data = data_frame[2].apply(lambda s: np.array(s.split(',')).astype(float)).values
    labels = data_frame[3].values
    labels = LabelEncoder().fit_transform(labels)
    # sample 
    idx = []
    for lbl in np.unique(labels):
        sample = np.random.choice(np.argwhere(labels==lbl).flatten(), sample_size)
        idx.extend(sample)
    return acc_data[idx], labels[idx]

def compare_methods(acc_data, labels, k, reps=3, cmp_type = cmp.CMP_MAX):
   
    #1) random partitions 
    score = []
    for i in range(reps):
        rnd_prt = np.random.randint(0, k, len(labels))
        rnd_prt = one_hot(rnd_prt)
        s = compare_partitions(labels, rnd_prt, method=cmp_type)
        score.append(s)
    print("Random: ", np.mean(score), np.std(score))
   
    #2) KMeans
    score = []
    for i in range(reps):
        kprt = KMeans(n_clusters=k).fit_predict(acc_data)
        kprt = one_hot(kprt)
        s = compare_partitions(labels, kprt, method=cmp_type)
        score.append(s)
    print("Kmeans: ", np.mean(score), np.std(score))

    #3) Matrix Factorization
    score = []
    for i in range(reps):
        mfprt = NMF(n_components=k).fit_transform(acc_data)
        s = compare_partitions(labels, mfprt, method=cmp_type)
        score.append(s)
    print("NMF: " , np.mean(score), np.std(score))

    print("Done!")

if __name__ == "__main__":
    data_file = os.path.join(DATA_PATH, "storks_2012_id_date_acc_behav_downsampledX30.csv")
    acc_data, labels = load(data_file)

    # print dist of labels 
    for lbl in np.unique(labels):
        print(lbl, np.where(labels==lbl)[0].shape[0] )
      
    # to features 
    acc_data = MultiScalePatches(scales=[3, 5, 7], size=20).fit(acc_data).compute(acc_data)
 
    # go! 
    for k in [5, 10, 20, 30]:
        print("Using k=", k)
        compare_methods(acc_data, labels, k, reps=3, cmp_type=cmp.CMP_MAX)
