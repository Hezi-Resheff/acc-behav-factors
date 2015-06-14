"""
Problem: we want to use the 3.33Hz obs for supervised learning of the new 10Hz data. 
Idea: downsample the new data... 
----
Here all we do is downsample and output to csv. The rest is done in AcceleRater. 

1) mV -> G 
2) 10Hz -> 3.33 Hz
3) 2csv
"""
import pandas as pd 
import numpy as np
import os

from Calibrator import Calibrator

def downsample(frame, source_hz, dest_hz):
    return frame

def calib_down_sample(calib_file, source_file, dest_file, source_hz=10, dest_hz=3.33):
    """Calibrate and downsample the data in source and output to dest"""         
    data = pd.DataFrame.from_csv(source_file, index_col='tag_id')       
    acc_data = data.apply(lambda row: pd.Series(np.array(row.acc_data.split(",")).astype(float)) , axis=1)
    
    data_downsampled = downsample(Calibrator(calib_file).transfom_all(acc_data), source_hz, dest_hz)    
    data_downsampled.to_csv(dest_file.format("ACC"), float_format="%.5f", index=False, header=False)

    acc_data.index = data.index
    data.drop('acc_data', axis=1, inplace=True)
    data_all = pd.concat((data, data_downsampled), axis=1)
    data_all.to_csv(dest_file.format("ALL"), float_format="%.5f")


if __name__ == "__main__":
    root_dir = "c:\\data\\Vultures"
    calib_file = os.path.join(root_dir, "params.csv")
    source_file = os.path.join(root_dir, "Vultures-10hz-10K-rand.csv")
    dest_file = os.path.join(root_dir, "Vultures-10hz-10K-rand__downsampled__{}.csv")
    calib_down_sample(calib_file, source_file, dest_file)





