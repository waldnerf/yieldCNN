import pandas as pd
import os, glob
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

import mysrc.constants as cst

def plot_nfolds_vs_correlation(df_, filename=''):
    plt.plot(df_.fold, df_.correlation, 'black', alpha=0.7)
    plt.xlabel('Number of folds')
    plt.ylabel('Correlation')
    plt.ylim([0.5, 1.05])
    plt.xticks(np.arange(1, df_.shape[0], step=2))
    if filename != '':
        plt.savefig(filename, dpi=350)
        plt.close()

def correlation_per_fold(df_):
    df_i_ = pd.DataFrame(np.zeros((df_['Year'].unique().shape[0], 3)),
                           columns=['id', 'fold', 'r2'])
    for i, year in enumerate(df_['Year'].unique()):
        idx = df_.Year[df_['Year'] == year].index.tolist()
        cv_pred = r2_score(df_['Observed'][idx], df_['Predicted'][idx])
        df_i_.iloc[i, :] = [cnt, i, cv_pred]
    df_i_['r2_pred'] = [np.mean(df_i_.r2[0:i]) for i in range(1, df_i_.shape[0] + 1)]
    df_i_['r2_true'] = float(df_i_.r2_pred.tail(1))
    df_i_ = df_i_.drop(columns=['r2'])
    return df_i_



dir_out = cst.my_project.params_dir
dir_res = dir_out / f'Archi_2DCNN_SISO_yield_norm/crop_0'


# this is the extension you want to detect
extension = '.csv'

df_out = pd.DataFrame()
cnt = 1
for root, dirs_list, files_list in os.walk(dir_res):
    for file_name in files_list:
        if (os.path.splitext(file_name)[-1] == extension) and ('_res_' in file_name):
            file_name_path = os.path.join(root, file_name)
            df = pd.read_csv(file_name_path)   # This is the full path of the filter file
            df_cv_i = correlation_per_fold(df)
            df_out = df_out.append(df_cv_i, ignore_index=False)
            cnt += 1
# Summarise per fold group
df_summary = df_out.groupby('fold').apply(lambda s: pd.Series({
    "correlation": s['r2_pred'].corr(s['r2_true'])
})).reset_index(level=0, drop=False)


plot_nfolds_vs_correlation(df_summary)