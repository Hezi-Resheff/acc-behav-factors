"""
Unsup using the patch-factorization method for the Pelican data, only where it is not flight.
Hope to find eating or something...
"""

import pandas as pd 
import numpy as np
from sklearn.decomposition import NMF
import os

# The  MultiScalePatches
from features import *   

ROOT = "c:\\data\\Ron\\NoFlight"

# convert to csv files... 
def convert_data_csv():    
    for f in  os.listdir(ROOT):
        pd.read_excel(os.path.join(ROOT, f), header=None).to_csv(os.path.join(ROOT, f).replace(".xlsx", ".csv"))
                
def process_data(k=10):
    # Load data
    print("Loading data")
    files = [f for f in  os.listdir(ROOT) if ".csv" in f]
    data = [pd.DataFrame.from_csv(os.path.join(ROOT, f)).dropna() for f in files]
  
    data_all = pd.concat(data).values
    index_all = ["{}_{}".format(fname, i) for fname, frame in zip(files, data) for i in frame.index.values]

    print("Training Feature")
    feature =  MultiScalePatches(scales=[4, 8], size=50, normalize_patches=True).fit(data_all).compute(data_all)

    print("Training Model")
    out = NMF(n_components=k).fit_transform(feature).argmax(axis=1)
       
    print("saving...")
    pd.DataFrame(out, index_all).to_csv(os.path.join(ROOT, "out" ,"out.csv"))
     
    print("Done!")
    
         
if __name__ == "__main__":
    #convert_data_csv()
    process_data()






