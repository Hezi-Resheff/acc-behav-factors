import numpy as np

smooth_kernel = np.array([-1, -1,  4, -1, -1])/5.0

def odba(sample):
    # -----------------------------------------
    # Calculate the ODBA of sample.
    # sample is X,Y,Z,X,Y,Z...
    # -----------------------------------------
    #brake up into X,Y,Z components
    X = sample[0::3]
    Y = sample[1::3]
    Z = sample[2::3]

     
    return np.absolute(np.convolve(X, smooth_kernel, mode="valid")).mean() + \
           np.absolute(np.convolve(Y, smooth_kernel, mode="valid")).mean() + \
           np.absolute(np.convolve(Z, smooth_kernel, mode="valid")).mean()
       
    
