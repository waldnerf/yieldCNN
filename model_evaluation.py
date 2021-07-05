import pandas as pd
import os, sys
import outputfiles.evaluation as ev
import mysrc.constants as cst


def weighed_average(grp):
    return grp._get_numeric_data().multiply(grp['Production'], axis=0).sum() / grp['Production'].sum()


def model_evaluation(fn_in, crop_ID, forecast_time, model_name, fn_asapID2AU, fn_stats90):
    """
    CNN2DFileFullPath: full path of the 2DCNN output, similar to mRRS but not identical
    asapID2AU_codeFileFullPath: full path to the cvs linking asap id to AU code
    stats90FullPath: full path to stats90 file (used in national yield computation weighted by production)
    crop_ID: the crop ID #
    """
    mRes = pd.read_csv(fn_in)
    # rename columns and replace ASAP ID with AU code
    mRes = mRes.rename(columns={"Observed": "yLoo_true", "Predicted": "yLoo_pred"})
    df_region = pd.read_csv(fn_asapID2AU)
    mRes = mRes.merge(df_region, left_on='ASAP1_ID', right_on='ASAP1_ID')
    mRes = mRes.drop(columns=['ASAP1_ID','AU_name'])

    error_AU_level = ev.allStats(mRes)
    meanAUR2 = ev.meanAUR2(mRes)  # equivalent to R2 within

    # National level stats
    # here I  weight the yield based on mean production
    # get AU mean production
    stats90 = pd.read_csv(fn_stats90, header=[0, 1])
    stats90.columns = pd.MultiIndex.from_tuples(stats90.columns)
    pd.MultiIndex.from_tuples(stats90.columns)
    tmp = stats90[stats90[('Crop_ID', 'Unnamed: 2_level_1')] == float(crop_ID)+1][[('Region_ID', 'Unnamed: 1_level_1'), ('Production', 'mean')]].droplevel(1, axis=1)
    mRes = pd.merge(mRes, tmp, how='left', left_on=['AU_code'], right_on=['Region_ID'])
    mCountryRes = mRes.groupby('Year')[['yLoo_pred', 'yLoo_true', 'Production']].apply(weighed_average).drop(
        ['Production'], axis=1)
    error_Country_level = ev.allStats_country(mCountryRes)

    # store results in dictionary
    outdict = {'runID': fn_in.name,
               'dataScaling': '',
               'DoOHEnc': '',
               'AddTargetMeanToFeature': '',
               'AddYieldTrend': '',
               'scoringMetric': '',
               'n_out_inner_loop': '',
               'nJobsForGridSearchCv': '',
               'Time': '',
               'Time_sampling': 'M',
               'forecast_time': forecast_time,
               'N_features': '',
               'N_OHE': '',
               'Features': '',
               'Ft_selection': '',
               'N_selected_fit': '',
               'Prct_selected_fit': '',
               'Selected_features_names_fit': '',
               'targetVar': 'Yield',
               'Crop':crop_ID+1,
               'Estimator': model_name,
               'Optimisation': '',
               'R2_f': '',  #TODO
               'R2_p': error_AU_level['Pred_R2'],
               'MAE_p': error_AU_level['Pred_MAE'],
               'rMAE_p': error_AU_level['rel_Pred_MAE'],
               'ME_p': error_AU_level['Pred_ME'],
               'RMSE_p': error_AU_level['Pred_RMSE'],
               'rRMSE_p': error_AU_level['rel_Pred_RMSE'],
               'HyperParGrid': '',  #TODO perhaps adding boundaries
               'HyperPar': '',  #TODO selected architecture details
               'avg_AU_R2_p(alias R2_WITHINp)': str(meanAUR2),
               'Country_R2_p': error_Country_level['Pred_R2'],
               'Country_MAE_p': error_Country_level['Pred_MAE'],
               'Country_ME_p': error_Country_level['Pred_ME'],
               'Country_RMSE_p': error_Country_level['Pred_RMSE'],
               'Country_rRMSE_p': error_Country_level['rel_Pred_RMSE'],
               'Country_FQ_rRMSE_p': error_Country_level['Pred_RMSE_FQ'],
               'Country_FQ_RMSE_p': error_Country_level['Pred_rRMSE_FQ'],
               'Mod_coeff': '',
               '%TimesPegged_left': '',
               '%TimesPegged_right': '',
               'run time (h)': ''
            }
    res_df = pd.DataFrame.from_dict([outdict])

    return res_df

if __name__ == "__main__":
    try:
        fn_indata = cst.my_project.data_dir / f'{cst.target}_full_2d_dataset.pickle'
        dir_out = cst.my_project.params_dir
        fn_asapID2AU = cst.root_dir / "raw_data" / "Algeria_REGION_id.csv"
        fn_stats90 = cst.root_dir / "raw_data" / "Algeria_stats90.csv"
        model_name = 'Archi+2DCNNw_MISO'
        out = []
        for crop_id in range(0, 3):
            for forecast_time in range(2, 9):
                dir_fn = dir_out / f'{model_name}/crop_{crop_id}/month_{forecast_time}/best_model'
                fn = [x for x in dir_fn.glob('*.csv')][0]
                res_i = model_evaluation(fn, crop_id, forecast_time, model_name, fn_asapID2AU, fn_stats90)
                out.append(res_i)

        df_out = pd.concat(out)
        df_out.to_csv(cst.root_dir / "data/full_model_evaluation.csv", index=False)

        print("0")
    except RuntimeError:
        print >> sys.stderr
        sys.exit(1)


# res = mRes2output(r'X:\PY_data\Histograms\example output file\archi-2DCNNw_MISO_res_68_test_35_5_2_1_0_0_2_2_2_sigmoid.csv',
#                   r'X:\PY_data\ML1_data_input\CountryStats\Algeria_REGION_id.csv',
#                   r'X:\PY_data\ML1_data_output\Algeria\Algeria_stats90.pkl',
#                   1)