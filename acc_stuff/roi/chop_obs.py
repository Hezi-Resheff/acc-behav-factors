"""
Chop up obs 2 4 sec segments.
"""

import pandas as pd 
import numpy as np
import os 
from sklearn.metrics import confusion_matrix

def chop(input_file):
    """keep random 40 sample sub-parts starting from an X measurement. format is X,X,X...Y,Y,Y...Z,Z,Z"""
    rows4s = []
    
    with open(input_file, "r") as data:
        for row in data:
            
            cells = row.split(',')
            vals, label = list(map(float, cells[:-1])), cells[-1].strip("\n")            

            nrows = len(vals) / 3
            if nrows <= 40:
             continue 
            
            # transform to X,Y,Z...
            vals = np.array(vals).reshape(3, nrows).T.ravel()
                        
            start_pnts = np.random.randint(0, nrows-40, 5)
            for sp in start_pnts:
                # select data
                d = vals[3*sp:3*sp+39]
                # transform back to X...Y...Z...
                d = d.reshape(13, 3).T.ravel()
                # save 
                rows4s.append(np.hstack((d,label)))
    
    return pd.DataFrame(rows4s)

def analyze_accelerater_out(input_file):
    data = pd.DataFrame.from_csv(input_file, header=None, index_col=None)
    print(np.unique(data[data.columns[-2]].values))
    print(confusion_matrix(data[data.columns[-2]].values, data[data.columns[-1]].values))


if __name__ == "__main__":
     root_dir = "C:\\Users\\t-yeresh\\data\\roi"
     
     #data4s = chop(os.path.join(root_dir, "vultures.csv" ))
     #data4s.to_csv(os.path.join(root_dir, "analize4s", "vlutures_4s.csv"), float_format="%.5f", index=False, header=False)

     analyze_accelerater_out(os.path.join(root_dir, "analize4s", "vlutures_4s_AcceleRater.csv"))
