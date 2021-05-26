import pickle
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
from toolz import interleave
from sklearn.model_selection import StratifiedShuffleSplit

import mysrc.constants as cst

import pandas as pd
from pathlib import Path


def get_season(date_, start_month, sep='-'):
    date_ = str(date_)
    date_s = date_.split(sep)
    if int(date_s[1]) < start_month:
        season = int(date_s[0]) - 1  # from previous season
    else:
        season = int(date_s[0])
    return season


def retain_regions(df, groups, key, target, cum=0.9):
    my_list = []

    for i in df[groups].unique():
        df_i = df.loc[df[groups] == i, :].sort_values(by=[key], ascending=False).copy().reset_index()
        df_i['key_pct'] = df_i.loc[:, key] / df_i[key].sum()
        df_i['key_cumsum'] = df_i['key_pct'].cumsum()
        idx = df_i.index[(df_i['key_cumsum'] < cum)].max()+1
        df_i = df_i.iloc[0:idx+1, :].copy()

        my_list = my_list + list(df_i[target])
    return list(set(my_list))

def main(fn_features, fn_stats,  step_dic, month_sos, fn_out=''):
    df_raw = pd.read_csv(fn_features)
    # Keep columns of interest
    df_raw = df_raw[['reg0_id', 'variable_name', 'date', 'mean']]
    df_raw = df_raw.rename(columns={'reg0_id': 'ASAP1_ID'}, inplace=False)

    # Add or modify column values
    df_raw['Year'] = df_raw['date'].apply(lambda x: get_season(x, month_sos))
    df_raw['step'] = df_raw['date'].str.rsplit(pat='-').apply(lambda x: str(x[1]) + '-' + str(x[2]))
    df_raw = df_raw[df_raw['step'].isin(step_dic.keys())]
    df_raw['step'] = df_raw['step'].apply(lambda x: step_dic[x])

    # Go from long to wide
    df_wide = df_raw.pivot_table(index=['ASAP1_ID', 'Year'],
                                 columns=['variable_name', 'step'],
                                 values='mean').dropna()
    df_wide.columns = df_wide.columns.map(lambda x: '{}_{}'.format(*x))
    df_wide.reset_index(inplace=True)

    df_stats = pd.read_csv(fn_stats)
    df_stats = df_stats[['Year', 'Area', 'Yield', 'Production', 'ASAP1_ID', 'Crop_name']].copy()
    df_stats['Crop_name'] = df_stats['Crop_name'].apply(lambda x: x.replace(' ', ''))
    # Get main producing regions
    df_filter = df_stats.groupby(['ASAP1_ID', 'Crop_name']).agg({'Production': 'mean'}).reset_index()
    region_ids = retain_regions(df_filter, groups='Crop_name', key='Production', target='ASAP1_ID')
    df_stats = df_stats.loc[df_stats['ASAP1_ID'].isin(region_ids), :].copy()


    df_statsw = df_stats.pivot_table(index=['ASAP1_ID', 'Year'],
                                     columns=['Crop_name'],
                                     values=['Area', 'Yield']).fillna(0)

    df_statsw.columns = df_statsw.columns.map(lambda x: '{}_{}'.format(*x))
    df_statsw.reset_index(inplace=True)

    # go to areal proportions
    my_cols = list(df_statsw.columns[df_statsw.columns.str.startswith('Area')])
    df_statsw.loc[:, my_cols] = df_statsw.loc[:, my_cols].apply(lambda x: x / x.sum(), axis=1)

    df_full = df_statsw.merge(df_wide, how='left')
    if fn_out != '':
        df_full.to_csv(fn_out, index=False)


if __name__ == "__main__":
    cst.root_dir = 'C:/Users/waldnfr/Documents/projects/leanyf'
    rdata_dir = Path(cst.root_dir, 'raw_data')

    step_dic = cst.step_dic
    month_sos = cst.month_sos

    fn_features = rdata_dir / f'{cst.target}_ASAP_data.csv'
    fn_stats = rdata_dir / f'{cst.target}_stats.csv'
    fn_out = cst.my_project.data_dir/ f'{cst.target}_full_dataset.csv'
    main(fn_features, fn_stats, step_dic, month_sos, fn_out)

