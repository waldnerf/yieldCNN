"""

"""
import dill as pickle
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
from toolz import interleave
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path

desired_width = 520
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

import mysrc.constants as cst
from outputfiles.plot import *


def get_season(date_, start_month, sep='-'):
    """
    Get season span from date
    """
    date_ = str(date_)
    date_s = date_.split(sep)
    if int(date_s[1]) < start_month:
        season = int(date_s[0]) - 1  # from previous season
    else:
        season = int(date_s[0])
    return season


def retain_regions(df, groups, key, target, cum=0.9):
    """
    Select ranked regions by decreasing order making up to 90% of the cumulative production
    """
    my_list = []

    for i in df[groups].unique():
        df_i = df.loc[df[groups] == i, :].sort_values(by=[key], ascending=False).copy().reset_index()
        df_i['key_pct'] = df_i.loc[:, key] / df_i[key].sum()
        df_i['key_cumsum'] = df_i['key_pct'].cumsum()
        idx = df_i.index[(df_i['key_cumsum'] < cum)].max() + 1
        df_i = df_i.iloc[0:idx + 1, :].copy()

        my_list = my_list + list(df_i[target])
    return list(set(my_list))


def get_2D_histogram(df, unit, year, ts_length, ts_start, normalise=True):
    """
    Convert dataframe into array
    """
    binDict = {
        'NDVI': {'min': 0.05, 'range': 0.85, 'n': 64},
        'rad': {'min': 40000, 'range': 280000, 'n': 64},
        'rainfall': {'min': 0, 'range': 100, 'n': 64},
        'temperature': {'min': -5, 'range': 50, 'n': 64}
    }

    df = df[df['ASAP1_ID'] == unit].copy()
    arr_out = []
    for var in binDict.keys():
        df_var = df[df['variable_name'] == var].copy()
        df_var['dates'] = pd.to_datetime(df_var['dekad'], format='%Y%m%d')
        df_var = df_var.sort_values(by='dates')
        xValues = df_var['dates'].tolist()
        xValues = [x.strftime("%Y%m%d") for x in xValues]
        binValues = np.linspace(binDict[var]['min'], binDict[var]['min'] + binDict[var]['range'],
                                num=binDict[var]['n'] + 1, endpoint=True)
        yValues = binValues[0:-1] + (binValues[1] - binValues[0]) / 2
        histo_cols = [col for col in df_var.columns if 'cls_cnt' in col]
        histo = df_var[histo_cols].to_numpy().transpose()

        start_sel = np.where([x == f'{year}{ts_start}' for x in xValues])[0][0]
        histo_year = histo[:, start_sel:(start_sel + ts_length)]
        if normalise:
            # normalise by pixel count per time step
            histo_sum = histo_year.sum(axis=0)
            # histo_min = np.zeros_like(histo_sum) #histo_year.min(axis=0)
            # histo_year = (histo_year - histo_min) / (histo_sum - histo_min)
            histo_year = histo_year / histo_sum
        arr_out.append(histo_year)

    arr_out = np.stack(arr_out, axis=2)
    return arr_out


def main(fn_features, fn_stats, fn_out='', normalise=True, save_plot=True):
    """
    Convert tabular data into arrays usable by a 1D CNN.

    Parameters
    ----------
    fn_features : str
        The file location of the input spreadsheet containing the satellite and climate data.
    fn_stats : str
        The file location of the spreadsheet containing the official statistics.
    fn_out : str, optional
        The location of the output file.
    normalise : bool, optional (default = True)
        Normalise the 2D histograms by image so that the minimum value is 0 and the maximum value is 1.
    save_plot : bool, optional (default = True)
        Save plots with input features.


    Returns
    -------
    None

    """
    df_stats = pd.read_csv(fn_stats)
    df_stats = df_stats[['Year', 'Area', 'Yield', 'Production', 'AU_name', 'ASAP1_ID', 'Crop_name']].copy()
    df_stats['Crop_name'] = df_stats['Crop_name'].apply(lambda x: x.replace(' ', ''))
    # Get main producing regions
    df_filter = df_stats.groupby(['ASAP1_ID', 'AU_name', 'Crop_name']).agg({'Production': 'mean'}).reset_index()
    region_ids = retain_regions(df_filter, groups='Crop_name', key='Production', target='ASAP1_ID')
    df_stats = df_stats.loc[df_stats['ASAP1_ID'].isin(region_ids), :].copy()

    df_statsw = df_stats.pivot_table(index=['ASAP1_ID', 'AU_name', 'Year'],
                                     columns=['Crop_name'],
                                     values=['Area', 'Yield']).fillna(0)

    df_statsw.columns = df_statsw.columns.map(lambda x: '{}_{}'.format(*x))
    df_statsw.reset_index(inplace=True)

    # Dropping 2001
    df_statsw = df_statsw.drop(df_statsw[df_statsw.Year == 2001].index)

    # go to area proportions
    my_cols = list(df_statsw.columns[df_statsw.columns.str.startswith('Area')])
    df_statsw.loc[:, my_cols] = df_statsw.loc[:, my_cols].apply(lambda x: x / x.sum(), axis=1)

    # -- Read and create 2D images
    df_raw = pd.read_csv(fn_features)
    df_raw = df_raw.rename(columns={"reg0_id": "ASAP1_ID"})

    # MM; NDVI of year 2001 starts in 10 01 while we need 08 01 for dtata augumentation
    # we mirror october into september and november into august
    # Sep:
    df_mirrored = df_raw.loc[(df_raw['variable_name'] == 'NDVI') & (df_raw['dekad'] == 20011001), :].copy()
    df_mirrored['dekad'] = 20010921
    df_raw = df_raw.append(df_mirrored)
    df_mirrored = df_raw.loc[(df_raw['variable_name'] == 'NDVI') & (df_raw['dekad'] == 20011011), :].copy()
    df_mirrored['dekad'] = 20010911
    df_raw = df_raw.append(df_mirrored)
    df_mirrored = df_raw.loc[(df_raw['variable_name'] == 'NDVI') & (df_raw['dekad'] == 20011021), :].copy()
    df_mirrored['dekad'] = 20010901
    # Aug:
    df_mirrored = df_raw.loc[(df_raw['variable_name'] == 'NDVI') & (df_raw['dekad'] == 20011101), :].copy()
    df_mirrored['dekad'] = 20010821
    df_raw = df_raw.append(df_mirrored)
    df_mirrored = df_raw.loc[(df_raw['variable_name'] == 'NDVI') & (df_raw['dekad'] == 20011111), :].copy()
    df_mirrored['dekad'] = 20010811
    df_raw = df_raw.append(df_mirrored)
    df_mirrored = df_raw.loc[(df_raw['variable_name'] == 'NDVI') & (df_raw['dekad'] == 20011121), :].copy()
    df_mirrored['dekad'] = 20010801
    df_raw = df_raw.append(df_mirrored)

    hists = []
    # Histograms with 4 variables
    variables = ['NDVI', 'Radiation', 'Rainfall', 'Temperature']
    for i, row in df_statsw.iterrows():
        # Start of season is at year -1 !!!
        hist = get_2D_histogram(df_raw, unit=int(row['ASAP1_ID']), year=int(row['Year']) - 1, ts_length=36,
                                ts_start='0801', normalise=normalise)
        hists.append(hist)

        # Plot data for each province-year
        super_title = f'{row["AU_name"]} ({row["Year"]}) - barley {round(row["Yield_Barley"], 2)} t/ha, ' \
                      f'soft wheat {round(row["Yield_Softwheat"], 2)} t/ha, ' \
                      f'durum wheat {round(row["Yield_Durumwheat"], 2)} t/ha, x0=1st dek Aug'
        fig_name = f'{row["AU_name"]}_{row["Year"]}_2Dinputs.png'
        if save_plot:
            plot_2D_inputs_by_region(hist, variables, super_title, fig_name=fig_name)
            plt.close()

    # Stacking and saving data
    hists = np.stack(hists, axis=0)
    if fn_out != '':
        # Saving the objects:
        with open(fn_out, 'wb') as f:
            pickle.dump({'stats': df_statsw, 'X': hists}, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    rdata_dir = Path(cst.root_dir, 'raw_data')
    fn_features = rdata_dir / f'{cst.target}_ASAP_2d_data.csv'
    fn_stats = rdata_dir / f'{cst.target}_stats.csv'
    save_plot = False
    normalise = True
    if normalise:
        fn_out = cst.my_project.data_dir / f'{cst.target}_full_2d_dataset_norm.pickle'
    else:
        fn_out = cst.my_project.data_dir / f'{cst.target}_full_2d_dataset_raw.pickle'
    main(fn_features, fn_stats, fn_out, normalise, save_plot)

# EOF
