"""
Reproducing results from Shay's (2015) manuscript without the supervised data, as part of a POC for the unsup method. 

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from local_settings import DATA_FOLDER, PLOTS_OUT
from feature import simple_features
from factors import FactorsSemiSup
from plots import plot_factors, plot_gps

def dates_selector(dates):
    month = pd.DatetimeIndex(dates).month
    good_dates = np.where((month >= 8) & (month <= 10))  # Aug-Oct is the Fall migration
    return good_dates

# load metadata
def load_meta():
    print("Loading metadata")
    return pd.DataFrame.from_csv(os.path.join(DATA_FOLDER, "storks_exp_rep", "meta.txt"), index_col=0)


# load migration dates table 1
def load_migration():
    print("Loading migration data")
    return pd.DataFrame.from_csv(os.path.join(DATA_FOLDER, "storks_exp_rep",
                                 "storks_migration_start_end_date.csv"), index_col=0).replace('\\N', np.NaN)


# load & process GPS data
def load_gps():
    print("Loading GPS")
    gps_data = pd.DataFrame.from_csv(os.path.join(DATA_FOLDER, "storks_exp_rep", "storks_mean_daily.csv"),
                                     header=None, index_col=None)
    gps_data.columns = ['id', 'date', 'lat', 'long']
    good_dates = dates_selector(gps_data.date)
    gps_data = gps_data.iloc[good_dates]
    return gps_data


# load & process ACC data
def load_acc():
    print("Loading ACC")
    acc_data = pd.DataFrame.from_csv(os.path.join(DATA_FOLDER, "storks_exp_rep", "storks_acc_flight.csv"),
                                     header=None, index_col=None)
    acc_data.columns = ['id', 'date', 'time'] + list(range(120)) + ['odba', 'behav']
    acc_data[list(range(119))] = acc_data[list(range(119))].apply(lambda v: [int(s.strip("\\")) for s in v]) # acc is val1\,val2\,...\,val120
    return acc_data

# translate raw ACC to A/P factors for a single animal 
def get_factors(acc):
    """ Translate the acc data into a factor representation """
    features_acc = simple_features().compute(acc[list(range(120))].values)
    feature_frame = pd.DataFrame(features_acc, index=acc.behav)
    factor_loading = FactorsSemiSup(labels=['AF', 'PF']).fit_transform(feature_frame)
    factor_loading.index = acc.date 
    return factor_loading 
    

def animal_plots(gps, acc, meta):
    print("Animal plots...")
    animals = acc.id.unique()
    for animal in animals:
        print(animal)
        # compute factor loadings
        factor_loading = get_factors(acc[acc.id == animal])
        factor_loading = factor_loading.groupby(factor_loading.index).mean() #average daily factor loading 
        # format GPS data for the plot
        this_gps = gps[gps.id == animal]
        this_gps.index = this_gps.date 
        this_gps = this_gps[['lat', 'long']]
        # plot this animal 
        plot_factors(factor_loading, None, this_gps, show_true_behav=False)
        plt.show()


def build_exp_groups(gps, acc, meta, mig):
    """ return the mean active/passive flight per animal per year with a Juv/Adult flag """ 
    print("Generating experiment groups...")
    
    # Loop over all animals 
    data = []
    animals = acc.id.unique()
    for animal in animals:
        print(animal)
        
        # is this a E pathway animal? Otherwise not interested...
        if animal not in meta.index or 'W' in meta.loc[animal]['mig_flyway']: 
            print('Discarding; Not in metadata or uses W mig flyway.')
            continue
        
        # get data for this animal
        this_acc = acc[acc.id == animal]
        factor_loading = get_factors(this_acc)
        this_gps = gps[gps.id == animal]
        
        # brake the rest into years and continue processing... 
        for year, acc_year in factor_loading.groupby(pd.DatetimeIndex(this_acc.date).year):
            
            # is it Adult/Juv
            start_year = int(meta.loc[animal]['date_start'].strip(' ')[:4])
            status = "Juv" if year==start_year else "Adult" 
            print("Year: ", year, status)

            # process start/end dates of mig.
            this_mig = mig[(mig.index == animal) & (mig.year == year)]
            if this_mig.shape[0] == 0 or (this_mig['end_date'].isnull().values and status == "Juv"):
                print("Discarding; Didn't finish migration or not in migration file. ")
                continue

            # find the index of the data to use
            year_ix = pd.DatetimeIndex(this_acc.date).year == year
            mig_ix = pd.DatetimeIndex(this_acc.date)[year_ix] > this_mig['start_date'].values[0]
            if ~this_mig['end_date'].isnull().values:
                mig_ix[pd.DatetimeIndex(this_acc.date)[year_ix] < this_mig['end_date'].values[0]] = False

            # compute measures and add row
            true_behav = (this_acc[year_ix][mig_ix]["behav"]==2).mean()
            unsup_behav =  acc_year[mig_ix].mean(axis=0).values.tolist()

            n_samples = len(acc_year)
            min_lat = this_gps[pd.DatetimeIndex(this_gps.date).year==year]['lat'].min()

            row = [animal, year, status] +  [true_behav] + unsup_behav + [n_samples, min_lat]
            print(row)
            data.append(row)

    return pd.DataFrame(data, columns=['id', 'year', 'status', 'AF_true_frac', 'AF_factor', 'PF_factor', 'N', 'min_lat'])


def run_experiment(name):
    """ Put the experiment all together... """
    mig = load_migration()
    gps = load_gps()
    acc = load_acc()
    meta = load_meta()
    frame = build_exp_groups(gps, acc, meta, mig)
    frame.to_csv(os.path.join(DATA_FOLDER, "storks_exp_rep", "out", name))
    

def results_plots(name, min_n=2000):
    
    frame = pd.DataFrame.from_csv(os.path.join(DATA_FOLDER, "storks_exp_rep", "out", name))
    frame = frame[frame.N >= min_n]

    # 0) Sig tests
    behav_semi_sup = frame.AF_factor
    juv_idx = frame.status == 'Juv'
    adult_idx = frame.status == 'Adult'
    print("semi: ", np.array([1, .5])*ttest_ind(behav_semi_sup[juv_idx], behav_semi_sup[adult_idx])) # [1, .5]*[t, p]
    behav_sup = frame.AF_true_frac
    print("sup: ", np.array([1, .5])*ttest_ind(behav_sup[juv_idx], behav_sup[adult_idx]))

    # 1) scatter for AF_true_frac & AF_factor 
    frame.plot(kind='scatter', x='AF_true_frac', y='AF_factor')
    plt.show()

    # 2) compare Juv/Adults with AF_factor
    m = frame.groupby(frame['status'])['AF_factor'].mean()
    s = frame.groupby(frame['status'])['AF_factor'].sem()
    m.plot(kind='bar', yerr=s)
    plt.show() 

    # 2) compare Juv/Adults with AF_true_frac
    m = frame.groupby(frame['status'])['AF_true_frac'].mean()
    s = frame.groupby(frame['status'])['AF_true_frac'].sem()
    m.plot(kind='bar', yerr=s)
    plt.show()


if __name__ == "__main__":
    name = "JuvAdltSimpleF__JuvFinished.csv"
    # run_experiment(name)
    results_plots(name)
    print("Done.")



