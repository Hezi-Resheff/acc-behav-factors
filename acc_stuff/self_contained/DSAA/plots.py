"""
Make & save the final plots used in the paper.
"""


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import os 

# plot format -- can put this in config file but that's too much work for so little change 
pd.options.display.mpl_style = 'default' 
mpl.rcParams['axes.facecolor'] = 'w'      # Give me back the white background!
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['lines.markeredgewidth'] = 1.0


DATA_PATH = "C:\\Users\\t-yeresh\\data\\storks2012"
TABLE_DATA_PATH = "C:\\Users\\t-yeresh\\Google Drive\\PhD\\Manuscripts\\DSAA15--journal version\\res"


def loss_plot(in_name, out_name, y_label):
    """ The 0-1 and log loss plolts; Data from .csv files comes from the log of main.py
    Make plots with no background, large text, large legend, large X maks -- LARGE EVERYTHING! 
    """
    f = pd.DataFrame.from_csv(os.path.join(TABLE_DATA_PATH, in_name))
    f.plot(style=['>:','<--','^-.', 's--', 'o-'])
    plt.xlabel("Number of clusters", fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.legend(loc = 'center left', bbox_to_anchor=(1.0, 0.5))
    plt.savefig(os.path.join(TABLE_DATA_PATH, out_name), bbox_inches="tight")
    plt.close()

def walking_plots():
    """ Data comes from raw ACC """ 
    pass

if __name__ == "__main__":
    loss_plot("0-1-loss.txt", "0-1.png", "0-1 loss")
    loss_plot("log-loss.txt", "log.png", "Log loss")