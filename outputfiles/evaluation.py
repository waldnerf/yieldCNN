import sklearn.metrics as metrics
import numpy as np

def mean_error_nan(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    nas = np.logical_or(np.isnan(y_true), np.isnan(y_pred))          #2021-4-30 nan treatement added
    return np.mean(np.array(y_pred[~nas]) - np.array(y_true[~nas]))

def r2_nan(x, y):
    # scikit r2_score is not resistant to nan
    x = np.array(x)
    y = np.array(y)
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    return metrics.r2_score(x[~nas], y[~nas])

def mean_absolute_error_nan(x, y):
    # scikit not resistant to nan
    x = np.array(x)
    y = np.array(y)
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    return metrics.mean_absolute_error(x[~nas], y[~nas])

def rmse_nan(x, y):
    # scikit not resistant to nan
    x = np.array(x)
    y = np.array(y)
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    return metrics.mean_squared_error(x[~nas], y[~nas], squared=False)

# def mean_rel_abs_error(w):
#     """Mean Absolute Percentage Error"""
#     # mean (|y(year,au)-y_pred(year,au)| / mean(y(year,au) * 100)
#     w = w.join(w.groupby('AU_code')['yLoo_true'].mean(), on='AU_code', rsuffix='_mean')
#
#     w['mean_rel_abs_error'] = ((w['yLoo_true' ] -w['yLoo_pred']).abs() /w['yLoo_true_mean' ] *100)
#     return ((w['yLoo_true' ] -w['yLoo_pred']).abs() /w['yLoo_true_mean' ] *100).mean()

def allStats_country(mRes):
    # At national level, not as average of held out year (cv folder)
    # mean true y used for normalization
    y_true = np.array(mRes['yLoo_true'])
    nas = np.isnan(y_true)
    avg_y_true = np.mean(y_true[~nas])
    res = {
        'Pred_R2': r2_nan(mRes['yLoo_true'], mRes['yLoo_pred']),
        #'Pred_R2': mRes.groupby('Year').apply(lambda x: r2_nan(x['yLoo_true'], x['yLoo_pred'])).reset_index(drop=True).mean(),
        'Pred_MAE': mean_absolute_error_nan(mRes['yLoo_true'], mRes['yLoo_pred']),
        # 'Pred_MAE': mRes.groupby('Year').apply(lambda x: mean_absolute_error_nan(x['yLoo_pred'], x['yLoo_true'])).reset_index(drop=True).mean(),
        'Pred_ME': mean_error_nan(mRes['yLoo_true'], mRes['yLoo_pred']),
        #'Pred_ME': mRes.groupby('Year').apply(lambda x: mean_error_nan(x['yLoo_true'], x['yLoo_pred'])).reset_index(drop=True).mean(),
        'Pred_RMSE': rmse_nan(mRes['yLoo_true'], mRes['yLoo_pred']),
        #'Pred_RMSE': mRes.groupby('Year').apply(lambda x: rmse_nan(x['yLoo_true'], x['yLoo_pred'])).reset_index(drop=True).mean(),
        'rel_Pred_MAE': mean_absolute_error_nan(mRes['yLoo_true'], mRes['yLoo_pred']) / avg_y_true * 100.0,
        #'rel_Pred_MAE': mRes.groupby('Year').apply(lambda x: mean_absolute_error_nan(x['yLoo_pred'], x['yLoo_true'])).reset_index(drop=True).mean() / avg_y_true * 100.0,
        'rel_Pred_RMSE': rmse_nan(mRes['yLoo_true'], mRes['yLoo_pred']) / avg_y_true * 100.0
        #'rel_Pred_RMSE': mRes.groupby('Year').apply(lambda x: rmse_nan(x['yLoo_true'], x['yLoo_pred'])).reset_index(drop=True).mean() / avg_y_true * 100.0
    }
    #compute RMSE on the poorest years (First Quantile lower 25 pecentile)
    mresFQ = mRes[mRes['yLoo_true'] <= mRes['yLoo_true'].quantile(0.25)]
    #res['Pred_RMSE_FQ'] = np.sqrt(metrics.mean_squared_error(mresFQ['yLoo_true'], mresFQ['yLoo_pred']))
    res['Pred_RMSE_FQ'] = rmse_nan(mresFQ['yLoo_true'], mresFQ['yLoo_pred'])
    res['Pred_rRMSE_FQ'] = res['Pred_RMSE_FQ'] / avg_y_true * 100.0
    # if (Compute_Pred_MrAE):
    #     # using the mean of the target value per AU, compute the % rmse
    #     res['Pred_MrAE'] = mean_rel_abs_error(mRes)
    return res

def allStats(mRes):
    # mean true y used for normalization
    y_true = np.array(mRes['yLoo_true'])
    nas = np.isnan(y_true)
    avg_y_true = np.mean(y_true[~nas])
    res = {
        #'Pred_R2': metrics.r2_score(mRes['yLoo_true'], mRes['yLoo_pred']),
        #'Pred_MAE': metrics.mean_absolute_error(mRes['yLoo_true'], mRes['yLoo_pred']),
        #'Pred_ME': mean_error_nan(mRes['yLoo_true'], mRes['yLoo_pred']),
        #'Pred_RMSE': np.sqrt(metrics.mean_squared_error(mRes['yLoo_true'], mRes['yLoo_pred'])),

        'Pred_R2': mRes.groupby('Year').apply(lambda x: r2_nan(x['yLoo_true'], x['yLoo_pred'])).reset_index(drop=True).mean(),
        'Pred_MAE': mRes.groupby('Year').apply(lambda x: mean_absolute_error_nan(x['yLoo_pred'], x['yLoo_true'])).reset_index(drop=True).mean(),
        'rel_Pred_MAE': mRes.groupby('Year').apply(lambda x: mean_absolute_error_nan(x['yLoo_pred'], x['yLoo_true'])).reset_index(drop=True).mean() / avg_y_true * 100.0,
        'Pred_ME': mRes.groupby('Year').apply(lambda x: mean_error_nan(x['yLoo_true'], x['yLoo_pred'])).reset_index(drop=True).mean(),
        'Pred_RMSE':  mRes.groupby('Year').apply(lambda x: rmse_nan(x['yLoo_true'], x['yLoo_pred'])).reset_index(drop=True).mean(),
        'rel_Pred_RMSE': mRes.groupby('Year').apply(lambda x: rmse_nan(x['yLoo_true'], x['yLoo_pred'])).reset_index(drop=True).mean() / avg_y_true * 100.0
    }
    # #compute RMSE on the poorest years (First Quantile lower 25 pecentile)
    # mresFQ = mRes[mRes['yLoo_true'] <= mRes['yLoo_true'].quantile(0.25)]
    # #res['Pred_RMSE_FQ'] = np.sqrt(metrics.mean_squared_error(mresFQ['yLoo_true'], mresFQ['yLoo_pred']))
    # res['Pred_RMSE_FQ'] = rmse_nan(mresFQ['yLoo_true'], mresFQ['yLoo_pred'])
    # res['Pred_rRMSE_FQ'] = res['Pred_RMSE_FQ'] / avg_y_true * 100.0
    # # if (Compute_Pred_MrAE):
    # #     # using the mean of the target value per AU, compute the % rmse
    # #     res['Pred_MrAE'] = mean_rel_abs_error(mRes)
    return res

def meanAUR2(mRes):
    # Compute the mean of the tempral R2 computed by AU
    def r2_au(g):
        x = g['yLoo_true']
        y = g['yLoo_pred']
        #return metrics.r2_score(g['yLoo_true'], g['yLoo_pred'])
        return r2_nan(x, y)

    res = mRes.groupby('AU_code').apply(r2_au)
    return res.mean()

