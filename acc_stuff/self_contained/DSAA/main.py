"""
This is the driver for all the experiments. 
Running this should reproduce all the reported results. 
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.mixture import GMM
import os

from features import *
from util import *

DATA_PATH = "C:\\Users\\t-yeresh\\data\\storks2012"


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
    if None != sample_size:
        idx = []
        for lbl in np.unique(labels):
            sample = np.random.choice(np.argwhere(labels==lbl).flatten(), sample_size)
            idx.extend(sample)
        return acc_data[idx], labels[idx]
    else:
        return acc_data, labels

def compare_methods(acc_data, labels, k, reps=3, cmp_type = cmp.CMP_MAX):
   
    out = pd.Series(index=["random", "uniform", "KMeans", "NNMF", "GMM"])

    print("Comparing with k=%i" % k)
    print("="*80)

    #1) random partitions -- hard
    score = []
    for i in range(reps):
        rnd_prt = np.random.randint(0, k, len(labels))
        rnd_prt = one_hot(rnd_prt)
        s = compare_partitions(labels, rnd_prt, method=cmp_type)
        score.append(s)
    print("Random: ", np.mean(score), np.std(score))
    out["random"] = np.mean(score)

    #2)uniform partitions
    score = []
    for i in range(reps):
        rnd_prt = np.ones((len(labels), k))*(1/k)
        s = compare_partitions(labels, rnd_prt, method=cmp_type)
        score.append(s)
    print("Uniform: ", np.mean(score), np.std(score))
    out["uniform"] = np.mean(score)

   
    #3) KMeans
    score = []
    for i in range(reps):
        kprt = KMeans(n_clusters=k).fit_predict(acc_data)
        kprt = one_hot(kprt)
        s = compare_partitions(labels, kprt, method=cmp_type)
        score.append(s)
    print("Kmeans: ", np.mean(score), np.std(score))
    out["KMeans"] = np.mean(score)


    #4) Matrix Factorization
    score = []
    for i in range(reps):
        mfprt = NMF(n_components=k).fit_transform(acc_data)
        s = compare_partitions(labels, mfprt, method=cmp_type)
        score.append(s)

    prt = return_partitions(labels, mfprt)
    prt.labels = ['0', '1', '2', '3', '4']
    prt = prt.groupby(prt.index).mean()
    print(prt.div(prt.sum(axis=1), axis=0)*100)

    print("NMF: " , np.mean(score), np.std(score))
    out["NNMF"] = np.mean(score)

    #5) GMM 
    score = []
    for i in range(reps):
        gmmprt = GMM(n_components=k).fit(acc_data).predict_proba(acc_data)
        s = compare_partitions(labels, gmmprt, method=cmp_type)
        score.append(s)
    print("GMM: " , np.mean(score), np.std(score))
    out["GMM"] = np.mean(score)

    return out 

if __name__ == "__main__":
    # make nice plots 
    pd.options.display.mpl_style = 'default'

    # load data 
    data_file = os.path.join(DATA_PATH, "storks_2012_id_date_acc_behav_downsampledX30.csv")
    acc_data, labels = load(data_file, sample_size=20000)

    # print dist of labels 
    for lbl in np.unique(labels):
        print(lbl, np.where(labels==lbl)[0].shape[0])
      
    # to features 
    #acc_data = MultiScalePatches(scales=[3, 7], size=30).fit(acc_data).compute(acc_data)
    acc_data = MultiScalePatches(scales=[3, 7], size=30, normalize_patches=True).fit(acc_data).compute(acc_data)

    # go!
    ks = np.arange(5, 45, 5)
    #ks = [5, 10, 45]
    data_01 = pd.DataFrame(index = ks, columns = ["random", "uniform", "KMeans", "NNMF", "GMM"])
    data_log = pd.DataFrame(index = ks, columns = ["random", "uniform", "KMeans", "NNMF", "GMM"])
    for k in ks:
        out_01 = compare_methods(acc_data, labels, k, reps=3, cmp_type=cmp.CMP_MAX)
        out_log = compare_methods(acc_data, labels, k, reps=3, cmp_type=cmp.CMP_LOG)
        data_01.loc[k] = out_01
        data_log.loc[k] = out_log

    print(data_01)
    data_01.plot(style="-x")
    plt.xlabel("Number of clusters", fontsize=20)
    plt.ylabel("0-1 loss", fontsize=20)
    plt.show()

    print(data_log)
    data_log.plot(style="-x")
    plt.xlabel("Number of clusters", fontsize=18)
    plt.ylabel("Log loss", fontsize=18)
    plt.show()