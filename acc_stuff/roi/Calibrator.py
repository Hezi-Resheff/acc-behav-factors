
import numpy as np 
import pandas as pd 

class Calibrator:
    # --------------------------------------------------------------------------
    # Class for dealing with calibration of acc data
    # --------------------------------------------------------------------------

    EARTH_G = 9.8

    def __init__(self, data_path, verbose=True):
        # ----------------------------------------------------        
        # load parameters
        # ----------------------------------------------------       
        self.verbose = verbose

        #load params
        self.params = pd.DataFrame.from_csv(data_path, header=None)

    def transform(self, data, tag_id):
        # ------------------------------------
        # transform 1 row of acc data
        # ------------------------------------
        if tag_id in self.params.index:        
            X_slope, Y_slope, Z_slope, X_zero, Y_zero, Z_zero = self.params.loc[tag_id]            
        else:
            if self.verbose:
                print "Calibrator: error while running with tag_id %i" % tag_id
            return np.zeros(len(data))
        
        data = data.values 
        data[0::3] = ( data[0::3] - X_zero ) * X_slope 
        data[1::3] = ( data[1::3] - Y_zero ) * Y_slope 
        data[2::3] = ( data[2::3] - Z_zero ) * Z_slope 
        return data 

    def transfom_all(self, data_frame):
        return data_frame.apply(lambda row: self.transform(row, tag_id=row.name), axis=1)

