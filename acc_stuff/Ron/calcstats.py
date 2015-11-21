"""
Calc the stats for acc-behav learner.
LEGACY CODE
"""

import numpy as np
from scipy import stats

#data formats
DATA_FORMAT_XYZXYZ = 0
DATA_FORMAT_XXYYZZ = 1
DATA_FORMAT_PRECOMPUTED = 2
DATA_FORMAT_UNIAXIS = 3

#statistics to calculate
def _meanX(dataX,dataY,dataZ): return dataX.mean()
def _meanY(dataX,dataY,dataZ): return dataY.mean()
def _meanZ(dataX,dataY,dataZ): return dataZ.mean()

def _stdX(dataX,dataY,dataZ):  return dataX.std() 
def _stdY(dataX,dataY,dataZ):  return dataY.std() 
def _stdZ(dataX,dataY,dataZ):  return dataZ.std()

def _skewX(dataX,dataY,dataZ): return stats.skew(dataX)
def _skewY(dataX,dataY,dataZ): return stats.skew(dataY)
def _skewZ(dataX,dataY,dataZ): return stats.skew(dataZ)

def _kurtX(dataX,dataY,dataZ): return stats.kurtosis(dataX)
def _kurtY(dataX,dataY,dataZ): return stats.kurtosis(dataY)
def _kurtZ(dataX,dataY,dataZ): return stats.kurtosis(dataZ)

def _maxX(dataX,dataY,dataZ): return dataX.max()
def _maxY(dataX,dataY,dataZ): return dataY.max()
def _maxZ(dataX,dataY,dataZ): return dataZ.max()

def _minX(dataX,dataY,dataZ): return dataX.min()
def _minY(dataX,dataY,dataZ): return dataY.min()
def _minZ(dataX,dataY,dataZ): return dataZ.min()

def _normX(dataX,dataY,dataZ): return np.linalg.norm(dataX)
def _normY(dataX,dataY,dataZ): return np.linalg.norm(dataY)
def _normZ(dataX,dataY,dataZ): return np.linalg.norm(dataZ)

def _covXY(dataX,dataY,dataZ): return np.cov(dataX, dataY)[0][1]
def _covXZ(dataX,dataY,dataZ): return np.cov(dataX, dataZ)[0][1]
def _covYZ(dataX,dataY,dataZ): return np.cov(dataY, dataZ)[0][1]
    
def _rXY(dataX,dataY,dataZ): return _covXY(dataX,dataY,dataZ) / (_stdX(dataX,dataY,dataZ)*_stdY(dataX,dataY,dataZ))
def _rXZ(dataX,dataY,dataZ): return _covXZ(dataX,dataY,dataZ) / (_stdX(dataX,dataY,dataZ)*_stdZ(dataX,dataY,dataZ))
def _rYZ(dataX,dataY,dataZ): return _covYZ(dataX,dataY,dataZ) / (_stdY(dataX,dataY,dataZ)*_stdZ(dataX,dataY,dataZ))


def _dbaX(X, Y, Z, winSize=10):
    Xsmooth = [ X[start:start+winSize].mean() for start in range(winSize//2, len(X)-winSize//2) ]
    x = np.absolute((X[winSize//2:len(X)-winSize//2] - Xsmooth)).mean()
    return x
    
def _dbaY(X,Y,Z,winSize=10):
    Ysmooth = [ Y[start:start+winSize].mean() for start in range(winSize//2,len(X)-winSize//2) ]
    y = np.absolute((Y[winSize//2:len(Y)-winSize//2] - Ysmooth)).mean()
    return y

def _dbaZ(X,Y,Z,winSize=10):
    Zsmooth = [ Z[start:start+winSize].mean() for start in range(winSize//2,len(X)-winSize//2) ]
    z = np.absolute((Z[winSize//2:len(Z)-winSize//2] - Zsmooth)).mean()
    return z

def _odba(X,Y,Z,winSize=10):
    return _dbaX(X,Y,Z,winSize) + _dbaY(X,Y,Z,winSize) + _dbaZ(X,Y,Z,winSize)

def _meandiffXY(dataX,dataY,dataZ): return (dataX - dataY).mean()
def _meandiffXZ(dataX,dataY,dataZ): return (dataX - dataZ).mean()
def _meandiffYZ(dataX,dataY,dataZ): return (dataY - dataZ).mean()

def _stddiffXY(dataX,dataY,dataZ): return (dataX - dataY).std()
def _stddiffXZ(dataX,dataY,dataZ): return (dataX - dataZ).std()
def _stddiffYZ(dataX,dataY,dataZ): return (dataY - dataZ).std()

#used later
def _isExtremum(i,vector):
    return ( vector[i] > max((vector[i-2],vector[i-1],vector[i+1],vector[i+2])) or vector[i] < min((vector[i-2],vector[i-1],vector[i+1],vector[i+2])) )

#we define the wave amplitude as the mean diff between max/min points
def _waveampX(dataX,dataY,dataZ):
    extremum = [dataX[i] for i in range(2,len(dataX)-2) if _isExtremum(i,dataX)]
    return np.mean( [ np.abs(extremum[i] - extremum[i+1]) for i in range(len(extremum)-1)] )
 
def _waveampY(dataX,dataY,dataZ):
    extremum = [dataY[i] for i in range(2,len(dataY)-2) if _isExtremum(i,dataY)]
    return np.mean( [ np.abs(extremum[i] - extremum[i+1]) for i in range(len(extremum)-1)] )

def _waveampZ(dataX,dataY,dataZ):
    extremum = [dataZ[i] for i in range(2,len(dataZ)-2) if _isExtremum(i,dataZ)]
    return np.mean( [ np.abs(extremum[i] - extremum[i+1]) for i in range(len(extremum)-1)] )

def _zeroCrossing(i,vec):
    return vec[i]*vec[i-1] < 0 

#the number of line crossings per sample (normalized per sample so that length doesn't matter)
def _crossingsXY(dataX,dataY,dataZ):
    diff = dataX-dataY
    return len( [i for i in range(1,len(diff)-1) if _zeroCrossing(i,diff)] ) / float(len(dataX))

def _crossingsXZ(dataX,dataY,dataZ):
    diff = dataX-dataZ
    return len( [i for i in range(1,len(diff)-1) if _zeroCrossing(i,diff)] ) / float(len(dataX))

def _crossingsYZ(dataX,dataY,dataZ):
    diff = dataY-dataZ
    return len( [i for i in range(1,len(diff)-1) if _zeroCrossing(i,diff)] ) / float(len(dataY))



def _25percentX(dataX,dataY,dataZ): return np.percentile(dataX,25)
def _25percentY(dataX,dataY,dataZ): return np.percentile(dataY,25)
def _25percentZ(dataX,dataY,dataZ): return np.percentile(dataZ,25)
def _50percentX(dataX,dataY,dataZ): return np.percentile(dataX,50)
def _50percentY(dataX,dataY,dataZ): return np.percentile(dataY,50)
def _50percentZ(dataX,dataY,dataZ): return np.percentile(dataZ,50)
def _75percentX(dataX,dataY,dataZ): return np.percentile(dataX,75)
def _75percentY(dataX,dataY,dataZ): return np.percentile(dataY,75)
def _75percentZ(dataX,dataY,dataZ): return np.percentile(dataZ,75)

_stat_functions = [ _meanX,_meanY,_meanZ,
                    _stdX,_stdY,_stdZ,
                    _skewX,_skewY,_skewZ,
                    _kurtX,_kurtY,_kurtZ,
                    _maxX,_maxY,_maxZ,
                    _minX,_minY,_minZ,
                    _normX,_normY,_normZ,
                    _covXY,_covXZ,_covYZ,
                    _rXY,_rXZ,_rYZ,
                    _dbaX,_dbaY,_dbaZ,_odba,
                    _meandiffXY,_meandiffXZ,_meandiffYZ,
                    _stddiffXY,_stddiffXZ,_stddiffYZ,
                    _waveampX,_waveampY,_waveampZ,
                    _crossingsXY,_crossingsXZ,_crossingsYZ,
                    _25percentX,_25percentY,_25percentZ,
                    _50percentX,_50percentY,_50percentZ,
                    _75percentX,_75percentY,_75percentZ
                  ]

statNames = ['MeanX','MeanY','MeanZ',
             'stdX','stdY','stdZ',
             'SkX','SkY','SxZ',
             'KuX','KuY','KuZ',
             'MaxX','MaxY','MaxZ',
             'MinX','MinY','MinZ',
             'normX','normY','normZ',
             'cov(x,y)','cov(x,z)','cov(y,z)',
             'r(x,y)','r(x,z)','r(y,z)',
             'DBA_X','DBA_Y','DBA_Z','ODBA',
             'mean-diff_XY','mean-diff_XZ','mean-diff_XZ',
             'std-diff_XY','std-diff_XZ','std-diff_YZ',
             'wave amplitude X','wave amplitude Y','wave amplitude Z',
             'line crossings XY','line crossings XZ','line crossings YZ',
             'X 25%','Y 25%','Z 25%',
             'X 50%','Y 50%','Z 50%',
             'X 75%','Y 75%','Z 75%']


"""
The list of statistics that can be calculated. 
"""
statFullNames = [ {'name':'Mean','three_ax':True,'numeric_code':[0,1,2]} ,
                  {'name':'Std','three_ax':True,'numeric_code':[3,4,5]} ,
                  {'name':'Skewness','three_ax':True,'numeric_code':[6,7,8]} ,
                  {'name':'Kurtosis','three_ax':True,'numeric_code':[9,10,11]} ,
                  {'name':'Max','three_ax':True,'numeric_code':[12,13,14]} ,
                  {'name':'Min','three_ax':True,'numeric_code':[15,16,17]} ,
                  {'name':'Norm','three_ax':True,'numeric_code':[18,19,20]} ,
                  {'name':'Cov','three_ax':True,'numeric_code':[21,22,23]} ,
                  {'name':'r','three_ax':True,'numeric_code':[24,25,26]} ,
                  {'name':'DBA','three_ax':True,'numeric_code':[27,28,29]} ,
                  {'name':'ODBA','three_ax':False,'numeric_code':[30]},
                  {'name':'mean-diff','three_ax':True,'numeric_code':[31,32,33]},
                  {'name':'std-diff','three_ax':True,'numeric_code':[34,35,36]},
                  {'name':'wave amplitude','three_ax':True,'numeric_code':[37,38,39]},
                  {'name':'line crossings','three_ax':True,'numeric_code':[40,41,42]},
                  {'name':'25%','three_ax':True,'numeric_code':[43,44,45]},
                  {'name':'50%','three_ax':True,'numeric_code':[46,47,48]},
                  {'name':'75%','three_ax':True,'numeric_code':[49,50,51]}
                  
                ]
    

                 
        
class stat_calculator():
    def __init__(self, what_to_calc=None, data_format=DATA_FORMAT_XYZXYZ):
        if what_to_calc == None:
            what_to_calc = list(range(len(_stat_functions)))
        self._mask = what_to_calc
        self._format = data_format
        
    def _calc_row(self,data):
        if self._format == DATA_FORMAT_XYZXYZ:
            dataX = data[0::3]
            dataY = data[1::3]
            dataZ = data[2::3]
        elif self._format == DATA_FORMAT_XXYYZZ:
            row_len = len(data) / 3  
            dataX = data[:row_len]
            dataY = data[row_len:2*row_len]
            dataZ = data[row_len*2:]
        elif self._format == DATA_FORMAT_PRECOMPUTED:
            return np.array(data)
        elif self._format == DATA_FORMAT_UNIAXIS:
            dataX = data
            dataY = None
            dataZ = None
        else:
            return None
       
        return [ _stat_functions[i](dataX, dataY, dataZ) for i in self._mask ]
    
    def get_stats(self,data):
        stats = np.array( [ self._calc_row( np.array(row).astype(float) ) for row in data ] )
        stats[np.isnan(stats)] = 0
        return stats

    def get_stats_labels(self, data):
        '''
        data has a label in the last index of each row.
        return: stats  - the array of stats for each row.
                labels - the list of the labels pealed off each row.     
        '''
        stats = np.array( [ self._calc_row( np.array(row[:-1]).astype(float) ) for row in data ] )
        stats[np.isnan(stats)] = 0
        labels = [ row[-1] for row in data ]
        return stats,labels 

        
        





    
