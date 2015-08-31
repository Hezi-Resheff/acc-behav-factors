"""
The tags are on a leg ring and rotate. The (self) X-Y plane needs to be treated as a single dim. 
"""

import pandas as pd 
from features import represent

def generate_features_4_accelerater(raw_data):
    ftr_frame = pd.DataFrame([represent(row, label=True) for row in raw_data.values])        
    return ftr_frame

if __name__ == "__main__":
    # ============ Storks
    # INPUT_FILE = "C:\\Users\\t-yeresh\\data\\storks2012\\obs.csv"
    # OUTPUT_FILE = "C:\\Users\\t-yeresh\\data\\storks2012\\obs_mix_xy.csv"
    
    # =========== Zoo data from Sasha -- Back data 
    # INPUT_FILE = "C:\\Users\\t-yeresh\\Downloads\\AcceleRaterDataBackSelected.csv"
    # OUTPUT_FILE = "C:\\Users\\t-yeresh\\Downloads\\AcceleRaterDataBackSelected__mix_xy.csv"
     
    # =========== Zoo data from Sasha -- Leg data 
    INPUT_FILE = "C:\\Users\\t-yeresh\\Downloads\\AcceleRaterDataLegSelected.csv"
    OUTPUT_FILE = "C:\\Users\\t-yeresh\\Downloads\\AcceleRaterDataLegSelected__mix_xz.csv"

    data = pd.DataFrame.from_csv(INPUT_FILE, header=False, index_col=None)
    generate_features_4_accelerater(data).to_csv(OUTPUT_FILE, header=None, index=False)

    