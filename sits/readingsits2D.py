#!/usr/bin/python

""" 
    Some functions to read and compute spectral features on SITS
"""

import sys, os
import numpy as np
import pandas as pd
import math
import random
import itertools
import dill as pickle
from tensorflow.keras.utils import to_categorical

import csv

# -----------------------------------------------------------------------
# ---------------------- SATELLITE MODULE
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
def data_reader(name_file):
    """
        Read the data contained in name_file
        INPUT:
            - name_file: file where to read the data
        OUTPUT:
            - X: variable vectors for each example
            - polygon_ids: id polygon (use e.g. for validation set)
            - Y: label for each example
    """

    with open(name_file, 'rb') as handle:
        d = pickle.load(handle)#, protocol=pickle.HIGHEST_PROTOCOL)
    df = d['stats']

    # time series
    Xt = d['X'].astype(dtype='float32')

    # region
    region_ids = df.ASAP1_ID.astype(dtype='uint16').values

    # Year
    years = df.Year.astype(dtype='int64').values

    # Area
    my_cols = list(df.columns[df.columns.str.startswith('Area')])
    area = df.loc[:, my_cols].astype(dtype='float32').values

    # Yields
    my_cols = list(df.columns[df.columns.str.startswith('Yield')])
    y = df.loc[:, my_cols].astype(dtype='float32').values

    # make sure that we do not keep entries with 0 ton/ha yields, moved to function optimise_so_@D..
    #yields_2_keep = ~(y <= 0).any(axis=1)

    #return Xt[yields_2_keep, :, :, :], area[yields_2_keep, :], region_ids[yields_2_keep], years[yields_2_keep], y[yields_2_keep, :]
    return Xt, area, region_ids, years, y


# -----------------------------------------------------------------------
def subset_data(Xt, region_ohe, y, subset_bool):
    return Xt[subset_bool, :, :, :], region_ohe[subset_bool, :], y[subset_bool]


# -----------------------------------------------------------------------
def computingMinMax(Xt, per=0):
    min_per_t = np.percentile(Xt, per, axis=(0, 1, 2))
    max_per_t = np.percentile(Xt, 100 - per, axis=(0, 1, 2))
    return min_per_t, max_per_t

# -----------------------------------------------------------------------
def normalizingData(X, min_per, max_per, back=False):
    if back:
        return X * (max_per - min_per) + min_per
    else:
        return (X - min_per) / (max_per - min_per)




# -----------------------------------------------------------------------
def extractValSet(X_train, polygon_ids_train, y_train, val_rate=0.1):
    unique_pol_ids_train, indices = np.unique(polygon_ids_train,
                                              return_inverse=True)  # -- pold_ids_train = unique_pol_ids_train[indices]
    nb_pols = len(unique_pol_ids_train)

    ind_shuffle = list(range(nb_pols))
    random.shuffle(ind_shuffle)
    list_indices = [[] for i in range(nb_pols)]
    shuffle_indices = [[] for i in range(nb_pols)]
    [list_indices[ind_shuffle[val]].append(idx) for idx, val in enumerate(indices)]

    final_ind = list(itertools.chain.from_iterable(list_indices))
    m = len(final_ind)
    final_train = int(math.ceil(m * (1.0 - val_rate)))

    shuffle_polygon_ids_train = polygon_ids_train[final_ind]
    id_final_train = shuffle_polygon_ids_train[final_train]

    while shuffle_polygon_ids_train[final_train - 1] == id_final_train:
        final_train = final_train - 1

    new_X_train = X_train[final_ind[:final_train], :, :]
    new_y_train = y_train[final_ind[:final_train]]
    new_X_val = X_train[final_ind[final_train:], :, :]
    new_y_val = y_train[final_ind[final_train:]]

    return new_X_train, new_y_train, new_X_val, new_y_val

# EOF
