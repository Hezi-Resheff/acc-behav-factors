""" What dates are not in the database? Ran over the behav output per tag. """

import os
import pandas as pd


ROOT_DIR = ""

def process_folder(folder):
    out = []
    files = os.listdir(folder)
    for file in files:
        data = pd.DataFrame.from_csv(os.path.join(folder, file))
        tag = {"id":  file.strip(".csv"), 
               "min": str(data.date.min()),
               "max": str(data.date.max()),
               "missing": "; ".join([str(d.date()) for d in pd.date_range(data.date.min(),  data.date.max()) if str(d.date()) not in data.date.values])
               }
        out.append(tag)
    return pd.DataFrame(out)


if __name__ == "__main__":
    f10 = process_folder(os.path.join(ROOT_DIR, "10"))
    f10.to_csv(os.path.join(ROOT_DIR, "10Hz__missing.csv"))
    f3 = process_folder(os.path.join(ROOT_DIR, "3"))
    f3.to_csv(os.path.join(ROOT_DIR, "3Hz__missing.csv"))

    
