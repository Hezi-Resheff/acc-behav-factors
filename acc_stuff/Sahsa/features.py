
import numpy as np
from scipy import stats

#statistics to calculate
def _meanX(dataX, dataZ): return dataX.mean()
def _meanZ(dataX, dataZ): return dataZ.mean()

def _stdX(dataX, dataZ):  return dataX.std() 
def _stdZ(dataX, dataZ):  return dataZ.std()

def _skewX(dataX, dataZ): return stats.skew(dataX)
def _skewZ(dataX, dataZ): return stats.skew(dataZ)

def _kurtX(dataX, dataZ): return stats.kurtosis(dataX)
def _kurtZ(dataX, dataZ): return stats.kurtosis(dataZ)

def _maxX(dataX, dataZ): return dataX.max()
def _maxZ(dataX, dataZ): return dataZ.max()

def _minX(dataX, dataZ): return dataX.min()
def _minZ(dataX, dataZ): return dataZ.min()

def _normX(dataX, dataZ): return np.linalg.norm(dataX)
def _normZ(dataX, dataZ): return np.linalg.norm(dataZ)

def _covXZ(dataX, dataZ): return np.cov(dataX,dataZ)[0][1]    
def _rXZ(dataX, dataZ): return _covXZ(dataX, dataZ) / (_stdX(dataX, dataZ)*_stdZ(dataX, dataZ))


def _dbaX(X, Z, winSize=5):
    Xsmooth = [ X[start:start+winSize].mean() for start in range(winSize//2, len(X)-winSize//2) ]
    x = np.absolute((X[winSize//2:len(X)-winSize//2] - Xsmooth)).mean()
    return x
    
def _dbaZ(X, Z, winSize=5):
    Zsmooth = [ Z[start:start+winSize].mean() for start in range(winSize//2,len(X)-winSize//2) ]
    z = np.absolute((Z[winSize//2:len(Z)-winSize//2] - Zsmooth)).mean()
    return z

def _odba(X, Z,winSize=5):
    return _dbaX(X, Z, winSize) + _dbaZ(X, Z, winSize)

def _meandiffXZ(dataX, dataZ): return (dataX - dataZ).mean()
def _stddiffXZ(dataX, dataZ): return (dataX - dataZ).std()

#used later
def _isExtremum(i, vector):
    return ( vector[i] > max((vector[i-2],vector[i-1],vector[i+1],vector[i+2])) or vector[i] < min((vector[i-2],vector[i-1],vector[i+1],vector[i+2])) )

#we define the wave amplitude as the mean diff between max/min points
def _waveampX(dataX, dataZ):
    extremum = [dataX[i] for i in range(2,len(dataX)-2) if _isExtremum(i,dataX)]
    return np.mean( [ np.abs(extremum[i] - extremum[i+1]) for i in range(len(extremum)-1)] )
 
def _waveampZ(dataX, dataZ):
    extremum = [dataZ[i] for i in range(2,len(dataZ)-2) if _isExtremum(i,dataZ)]
    return np.mean( [ np.abs(extremum[i] - extremum[i+1]) for i in range(len(extremum)-1)] )

def _zeroCrossing(i,vec):
    return vec[i]*vec[i-1] < 0 

def _crossingsXZ(dataX, dataZ):
    diff = dataX-dataZ
    return len( [i for i in range(1,len(diff)-1) if _zeroCrossing(i,diff)] ) / float(len(dataX))

def _25percentX(dataX, dataZ): return np.percentile(dataX,25)
def _25percentZ(dataX, dataZ): return np.percentile(dataZ,25)
def _50percentX(dataX, dataZ): return np.percentile(dataX,50)
def _50percentZ(dataX, dataZ): return np.percentile(dataZ,50)
def _75percentX(dataX, dataZ): return np.percentile(dataX,75)
def _75percentZ(dataX, dataZ): return np.percentile(dataZ,75)

feature_functions =  [ 
    _meanX, _meanZ,
    _stdX, _stdZ,
    _skewX, _skewZ,
    _kurtX, _kurtZ,
    _maxX, _maxZ,
    _minX, _minZ,
    _normX, _normZ,
    _covXZ,
    _rXZ,
    _dbaX, _dbaZ, _odba,
    _meandiffXZ,
    _stddiffXZ,
    _waveampX, _waveampZ,
    _crossingsXZ,
    _25percentX, _25percentZ,
    _50percentX, _50percentZ,
    _75percentX, _75percentZ
]        
         
def represent(row, label=True):
    if label:
        l = row[-1]
        row = row[:-1]

    dataX = row[0::3].copy()
    dataY = row[1::3].copy()
    dataZ = row[2::3].copy()
    dataXY = (dataX**2 + dataY**2)**.5

    # For the leg data 
    # dataXY = (dataX**2 + dataZ**2)**.5
    # dataZ = dataY
    
    if label:
        return np.array([f(dataXY, dataZ) for f in feature_functions]+[l])
    else:    
        return np.array([f(dataXY, dataZ) for f in feature_functions])




    
