"""
Michele, 02/09/2021, bugs fix and merge preprocessing 1D and 2D
bugs:
!D and 2D:
-	Uses a different method to select 90% production regions  (union of crops) but then uses stats90 in model evaluation (some regions will be thus missing)
2D:
-	image norm was very strange (norm by cum with effect of having 1s in meteo var and no 1s in NDVI), now max norm and outside preprocessing
-	NDVI histograms had stripes due to interaction between bins and NDVI sampling, corrected
1D:
-	Preprocessing was wrong (data of 6 masks were kept and averaged!, wrong year of yield was associated)
-	Data were staring in October while for the 2D in sep


Convert tabular data into arrays usable by a 1D CNN.
"""

import argparse
import pandas as pd
from pathlib import Path
import mysrc.constants as cst
from outputfiles.plot import *
from sits.readingsits2D import *

desired_width = 520
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

def get_YieldSeason(date_, start_month, sep='-'):
    """
    Get season span from date
    """
    date_ = str(date_)
    date_s = date_.split(sep)
    if int(date_s[1]) < start_month:
        season = int(date_s[0]) # from previous season
    else:
        season = int(date_s[0]) + 1
    return season

def get_2D_histogram(df, unit, year, ts_length, ts_start):
    """
    Convert dataframe into array
    """
    # binDict_v2 = {
    #     'NDVI': {'min': 0.05, 'range': 0.85, 'n': 64},
    #     'rad': {'min': 40000, 'range': 280000, 'n': 64},
    #     'rainfall': {'min': 0, 'range': 100, 'n': 64},
    #     'temperature': {'min': -5, 'range': 50, 'n': 64}
    # }
    #M: This version 3 fixing the issue of NDVI stripes due to interaction between bin size and NDVI coding 8 bit TODO: source from constant
    binDict = {
        'NDVI': {'min': 0.05, 'range': 0.9216, 'n': 64},
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

        arr_out.append(histo_year)

    arr_out = np.stack(arr_out, axis=2)
    return arr_out

def main(D, fn_features, fn_stats, fn_stats90, fn_out='', save_plot=True):
    # process official stats
    # get the list of 90% regions
    df_stats90 = pd.read_csv(fn_stats90, header=[0, 1])
    df_stats90.columns = pd.MultiIndex.from_tuples(df_stats90.columns)
    df_stats90 = df_stats90[[('Region_ID', 'Unnamed: 1_level_1'), ('ASAP1_ID', 'first'), ('Crop_ID', 'Unnamed: 2_level_1'), ('AU_name', 'first'), ('Production', 'mean')]].droplevel(1, axis=1)
    # keep only those from official stats
    df_stats = pd.read_csv(fn_stats)
    df_stats = df_stats[['Year', 'Area', 'Yield', 'Production', 'ASAP1_ID', 'AU_name', 'Crop_ID', 'Crop_name']].copy()
    df_stats['Crop_name'] = df_stats['Crop_name'].apply(lambda x: x.replace(' ', ''))
    df_stats2use = df_stats.copy()
    df_stats2use = df_stats2use[0: 0]
    # Get main producing regions
    cropsIDs = df_stats90['Crop_ID'].unique()
    for c in cropsIDs:
        regions2keep = df_stats90[df_stats90["Crop_ID"] == c]["ASAP1_ID"].unique()
        tmp = df_stats[df_stats['Crop_ID'] == c].copy()
        df_stats2use = df_stats2use.append(tmp[tmp["ASAP1_ID"].isin(regions2keep)])
    df_statsw = df_stats2use.pivot_table(index=['ASAP1_ID', 'AU_name', 'Year'],
                                     columns=['Crop_name'],
                                     values=['Area', 'Yield']).fillna(0)

    df_statsw.columns = df_statsw.columns.map(lambda x: '{}_{}'.format(*x))
    df_statsw.reset_index(inplace=True)
    # Drop 2001
    df_statsw = df_statsw.drop(df_statsw[df_statsw.Year == 2001].index)
    # After this Franz was going to area proportion (in preprocess_1D and preprocess_2D) but don't know why and in a way that does not make sense. I removed it

    # process features
    df_raw = pd.read_csv(fn_features)
    df_raw = df_raw.rename(columns={"reg0_id": "ASAP1_ID"})
    if D == 1:
        # make sure taking only crop static mask
        df_raw = df_raw[(df_raw['class_name'] == 'crop') & (df_raw['classset_name'] == 'static masks')]
        # Keep columns of interest
        df_raw = df_raw[['ASAP1_ID', 'variable_name', 'date', 'mean']]
        # NDVI of year 2001 starts in 10 01 while we need 08 01 for data augmentation
        # we mirror october into september and november into august
        # Sep:
        df_mirrored = df_raw.loc[(df_raw['variable_name'] == 'NDVI') & (df_raw['date'] == '2001-10-01'), :].copy()
        df_mirrored['date'] = '2001-09-21'
        df_raw = df_raw.append(df_mirrored)
        df_mirrored = df_raw.loc[(df_raw['variable_name'] == 'NDVI') & (df_raw['date'] == '2001-10-11'), :].copy()
        df_mirrored['date'] = '2001-09-11'
        df_raw = df_raw.append(df_mirrored)
        df_mirrored = df_raw.loc[(df_raw['variable_name'] == 'NDVI') & (df_raw['date'] == '2001-10-21'), :].copy()
        df_mirrored['date'] = '2001-09-01'
        df_raw = df_raw.append(df_mirrored)
        # Aug:
        df_mirrored = df_raw.loc[(df_raw['variable_name'] == 'NDVI') & (df_raw['date'] == '2001-11-01'), :].copy()
        df_mirrored['date'] = '2001-08-21'
        df_raw = df_raw.append(df_mirrored)
        df_mirrored = df_raw.loc[(df_raw['variable_name'] == 'NDVI') & (df_raw['date'] == '2001-11-11'), :].copy()
        df_mirrored['date'] = '2001-08-11'
        df_raw = df_raw.append(df_mirrored)
        df_mirrored = df_raw.loc[(df_raw['variable_name'] == 'NDVI') & (df_raw['date'] == '2001-11-21'), :].copy()
        df_mirrored['date'] = '2001-08-01'
        df_raw = df_raw.append(df_mirrored)
        # first month is August
        first_month = cst.first_month_in__raw_data #first month after mirroring
        # We make the year the one of the yield
        df_raw['Year'] = df_raw['date'].apply(lambda x: get_YieldSeason(x, first_month))
        df_raw['dek'] = df_raw['date'].str.rsplit(pat='-').apply(lambda x: str(x[1]) + '-' + str(x[2]))
        df_raw['dek'] = df_raw['dek'].apply(lambda x: cst.MMDD2dek_dict[x])
        # and step is the dekad starting (=1) at the first calendar dekad to be considered
        df_raw['step'] = df_raw['dek']
        def replace(x, dek2be1):
            if x >= dek2be1:
                x = x - dek2be1 + 1
            else:
                x = 36 - dek2be1 + 1 + x
            return x
        df_raw['step'] = df_raw['step'].apply( lambda x: replace(x, (first_month-1)*3+1))
        counts = df_raw.groupby(['ASAP1_ID', 'variable_name','date']).size().reset_index(name='n')
        duplicates = counts[counts['n'] > 1]
        if not duplicates.empty:
            print('Duplicated detected in 1D input data')
            print(duplicates)
            sys.exit
        df_wide = df_raw.pivot_table(index=['ASAP1_ID', 'Year'], columns=['variable_name', 'step'], values='mean')
        df_wide.columns = df_wide.columns.map(lambda x: '{}_{}'.format(*x))
        df_wide.reset_index(inplace=True)
        df_full = df_statsw.merge(df_wide, how='left')
        if fn_out != '':
            df_full.to_csv(Path(fn_out).with_suffix('.csv'), index=False)
            with open(Path(fn_out).with_suffix('.pickle'), 'wb') as f:
                pickle.dump(df_full, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif D == 2:
        # NDVI of year 2001 starts in 10 01 while we need 08 01 for data augmentation
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
                                    ts_start='0801')
            hists.append(hist)
            if save_plot:
                # Plot data for each province-year
                super_title = f'{row["AU_name"]} ({row["Year"]}) - barley {round(row["Yield_Barley"], 2)} t/ha, ' \
                              f'soft wheat {round(row["Yield_Softwheat"], 2)} t/ha, ' \
                              f'durum wheat {round(row["Yield_Durumwheat"], 2)} t/ha, x0=1st dek Aug'
                fig_name = cst.my_project.root_dir / "figures" / f'{row["AU_name"]}_{row["Year"]}_raw_2Dinputs.png'
                plot_2D_inputs_by_region(hist, variables, super_title, fig_name=fig_name)
                plt.close()

        # Stacking and saving data
        hists = np.stack(hists, axis=0)
        if fn_out != '':
            # Saving the objects:
            with open(Path(fn_out).with_suffix('.pickle'), 'wb') as f:
                pickle.dump({'stats': df_statsw, 'X': hists}, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # ---- Define parser
    parser = argparse.ArgumentParser(description='Input reader')
    parser.add_argument('--D', type=int, default=1,
                        help='Dimension of inputs, can be 1D time series or 2D histograms')
    parser.add_argument('--saveplot', type=bool, default=False,
                        help='Save the plots')
    args = parser.parse_args()
    # ---- Get parameters
    D = args.D
    save_plot = args.saveplot
    print(f'D = {D}')


    rdata_dir = Path(cst.root_dir, 'raw_data')
    fn_stats = rdata_dir / f'{cst.target}_stats.csv'
    fn_stats90 = rdata_dir / f'{cst.target}_stats90.csv'

    if D == 1:
        fn_features = rdata_dir / f'{cst.target}_ASAP_data.csv'
        fn_out = cst.my_project.data_dir / f'{cst.target}_full_1d_dataset_raw'#.csv' #TODO: use nicer names
    elif D == 2:
        fn_features = rdata_dir / f'{cst.target}_ASAP_2d_data_v3.csv'  # Algeria_ASAP_2d_data_v3
        fn_out = cst.my_project.data_dir / f'{cst.target}_full_2d_dataset_raw'#.pickle'

    main(D, fn_features, fn_stats, fn_stats90, fn_out, save_plot)