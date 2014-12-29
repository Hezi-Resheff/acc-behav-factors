"""
Plots util. just for fun. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import gridspec
pd.options.display.mpl_style = 'default'

def plot_factors(factor, behav, gps, where_behav=None, show_true_behav=True):
    
     if where_behav:
         # only plot for a single true behav
         rows = np.where(behav[where_behav].sum(axis=1) )[0]
         behav = behav.ix[rows][where_behav]
         factor = factor.ix[rows]
     
     if show_true_behav:
         gs = gridspec.GridSpec(3, 1, height_ratios=[2, 4, 1])    
     else:
         gs = gridspec.GridSpec(2, 1, height_ratios=[2, 4])    
            
        
     ax0 = plt.subplot(gs[0])
     gps.plot(ax=ax0, style="-x", title="GPS")
     ax0.xaxis.set_visible(True)

     # factor loadings 
     ax1 = plt.subplot(gs[1], sharex=ax0)
     factor.plot(ax=ax1, style="-x", title="Factor Loadings")
     
     # behav 
     if show_true_behav:
         ax2 = plt.subplot(gs[2], sharex=ax0)
         behav.plot(ax=ax2, style=".", title="Sup Behav")
         ax2.set_ylim(.9, 1.1)
         ax2.legend_.remove()
         ax2.yaxis.set_visible(False)

     

