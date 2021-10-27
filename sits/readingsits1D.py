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

    data = pd.read_csv(name_file)

    # time series
    my_cols = [any(x) for x in zip(data.columns.str.startswith('NDVI'),
                                   data.columns.str.startswith('rad'),
                                   data.columns.str.startswith('rain'),
                                   data.columns.str.startswith('temp'))
               ]
    Xt = data.loc[:, my_cols].astype(dtype='float32').values

    # region
    region_ids = data.ASAP1_ID.astype(dtype='uint16').values

    # Year
    years = data.Year.astype(dtype='int64').values

    # Area
    my_cols = list(data.columns[data.columns.str.startswith('Area')])
    Xv = data.loc[:, my_cols].astype(dtype='float32').values

    # Yields
    my_cols = list(data.columns[data.columns.str.startswith('Yield')])
    y = data.loc[:, my_cols].astype(dtype='float32').values

    return Xt, Xv, region_ids, years, y





def subset_data(Xt, region_ohe, y, subset_bool):
    return Xt[subset_bool, :], region_ohe[subset_bool, :], y[subset_bool]

# -----------------------------------------------------------------------
def reshape_data(X, nchannels):
    """
        Reshaping (feature format (3 bands): d1.b1 d1.b2 d1.b3 d2.b1 d2.b2 d2.b3 ...)
        INPUT:
            -X: original feature vector ()
            -nchannels: number of channels
        OUTPUT:
            -new_X: data in the good format for Keras models
    """
    return X.reshape(X.shape[0], int(X.shape[1] / nchannels), nchannels, order='F')

# -----------------------------------------------------------------------
def computingMinMax(x, per=2):
    min_per = np.percentile(x, per, axis=(0, 1))
    max_per = np.percentile(x, 100 - per, axis=(0, 1))
    return min_per, max_per

# -----------------------------------------------------------------------
def read_minMaxVal(minmax_file):
    with open(minmax_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        min_per = next(reader)
        max_per = next(reader)
    min_per = [float(k) for k in min_per]
    min_per = np.array(min_per)
    max_per = [float(k) for k in max_per]
    max_per = np.array(max_per)
    return min_per, max_per


# -----------------------------------------------------------------------
def save_minMaxVal(minmax_file, min_per, max_per):
    with open(minmax_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(min_per)
        writer.writerow(max_per)


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
