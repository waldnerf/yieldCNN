#!/usr/bin/python

import os
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import optuna
from optuna.samplers import TPESampler
import joblib
import random
import wandb
import global_variables

random.seed(4)

from deeplearning.architecture_cv_v3 import cv_Model
from deeplearning.architecture_complexity_1D import Archi_1DCNN, Archi_simple#Archi_1DCNN_MISO, Archi_1DCNN_SISO
from outputfiles import plot as out_plot
from outputfiles import save as out_save
from evaluation import model_evaluation as mod_eval
from sits import readingsits1D, common_functions_1D2D
import mysrc.constants as cst
import datetime

# global vars
N_CHANNELS = 4  # -- NDVI, Rad, Rain, Temp
dict_train_params = {
    'optuna_metric': 'rmse', #'rmse' or 'r2'
    #'N_EPOCHS': 200, # 100, #70,
    #'BATCH_SIZE': 128, #128
    'N_TRIALS': 100,
    #'lr': 0.01, #0.001 is Adam default
    'beta_1': 0.9, #all defaults
    'beta_2': 0.999,
    'decay':  0.01
}

# global vars - used in objective_2DCNN
model_type = None
Xt = None
Xtk = None
region_ohe = None
#dir_tgt = None
groups = None
#data_augmentation = None
#generator = None
y = None
crop_n = None
region_id = None
xlabels = None
ylabels = None
out_model = None
# to be used here and in architecture_cv


def main():
    starttime = datetime.datetime.now()
    # ---- Define parser
    parser = argparse.ArgumentParser(description='Optimise 1D CNN for yield and area forecasting')
    parser.add_argument('--model', type=str, default='1DCNN_MISO',
                        help='Model type: Single input single output (SISO) or Multiple inputs/Single output (MISO)')
    parser.add_argument('--target', type=str, default='yield', choices=['yield', 'area'], help='Target variable')
    parser.add_argument('--wandb', dest='wandb', action='store_true', default=False, help='Store results on wandb.io')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', default=False,
                        help='Overwrite existing results')
    args = parser.parse_args()

    # ---- Get parameters
    global model_type
    model_type = args.model
    if args.wandb:
        print('Wandb log requested')

    # ---- Define some paths to data
    fn_indata = str(cst.my_project.data_dir / f'{cst.target}_full_1d_dataset_raw.csv')
    print("Input file: ", os.path.basename(str(fn_indata)))

    fn_asapID2AU = cst.root_dir / "raw_data" / "Algeria_REGION_id.csv"
    fn_stats90 = cst.root_dir / "raw_data" / "Algeria_stats90.csv"

    # ---- Downloading
    Xt_full, area_full, region_id_full, groups_full, yld_full = readingsits1D.data_reader(fn_indata)
    # Once and for all
    # Change here format of Xt_full to keras (n_sample, n_deks, n_channels)
    Xtk_full = readingsits1D.reshape_data(Xt_full, N_CHANNELS)
    # loop through all crops
    global crop_n
    for crop_n in [2]: # range(y.shape[1]): #!TODO: now only 2 soft wheat (0 - Barley, 1 - Durum, 2- Soft)
        # clean trial history for a new crop
        trial_history = []

        # make sure that we do not keep entries with 0 ton/ha yields, each region has yield 0 even if it is not with 90% production
        yields_2_keep = ~(yld_full[:, crop_n] <= 0)
        #TC
        Xt_nozero = Xt_full[yields_2_keep, :]
        Xtk_nozero = Xtk_full[yields_2_keep,:,:]
        area = area_full[yields_2_keep, :]
        global region_id, groups
        region_id = region_id_full[yields_2_keep]
        groups = groups_full[yields_2_keep]
        yld = yld_full[yields_2_keep, :]    #TODO: keep only current crop
        # ---- Format target variable
        global y, xlabels, ylabels
        if args.target == 'yield':
            y = yld
            xlabels = 'Predictions (t/ha)'
            ylabels = 'Observations (t/ha)'
        elif args.target == 'area':
            y = area
            xlabels = 'Predictions (%)'
            ylabels = 'Observations (%)'

        # ---- Convert region to one hot
        global region_ohe
        region_ohe = common_functions_1D2D.add_one_hot(region_id)

        # loop by month
        for month in [7,4]: #TODO put back to all -> range(1, cst.n_month_analysis + 1)
            # ---- output files and dirs
            dir_out = cst.my_project.params_dir
            dir_out.mkdir(parents=True, exist_ok=True)
            dir_res = dir_out / f'Archi_{str(model_type)}_{args.target}'
            dir_res.mkdir(parents=True, exist_ok=True)
            global out_model
            out_model = f'{model_type}-{args.target}.h5'
            # crop dirs
            dir_crop = dir_res / f'crop_{crop_n}'
            dir_crop.mkdir(parents=True, exist_ok=True)
            # month dirs
            #global dir_tgt
            global_variables.dir_tgt = dir_crop / f'month_{month}'
            global_variables.dir_tgt.mkdir(parents=True, exist_ok=True)

            if (len([x for x in global_variables.dir_tgt.glob('best_model')]) != 0) & (args.overwrite is False):
                pass
            else:
                out_save.rm_tree(global_variables.dir_tgt)
                indices = list(range(0, Xt_full.shape[1] // N_CHANNELS))
                #
                first_month_in__raw_data = 8  # August; this is taken to allow data augmentation (after mirroring Oct and Nov of 2001 to Sep and Aug, all raw data start in August)
                # data are thus ordered according to a local year having index = 0 at first_month_in__raw_data
                first = (cst.first_month_input_local_year) * 3
                last = (cst.first_month_analysis_local_year + month - 1) * 3
                msel = [True if ((x >= first) and (x < last)) else False for x in indices] * N_CHANNELS
                # msel = [True if x < (month * 3) else False for x in indices] * N_CHANNELS
                ##TC
                global Xt
                global Xtk
                Xt = Xt_nozero[:, msel]
                Xtk = Xtk_nozero[:,first:last,:]
                print('------------------------------------------------')
                print('------------------------------------------------')
                print(f"")
                print(f'=> noarchi: {model_type}'
                      f' {args.target} - crop: {crop_n} - month: {month}')
                if dict_train_params['optuna_metric']=='rmse':
                    dirct = 'minimize'
                elif dict_train_params['optuna_metric']=='r2':
                    dirct = 'maximize'

                study = optuna.create_study(direction=dirct,
                                            sampler=TPESampler(seed=10),
                                            pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=6)
                                            )

                # Force the sampler to sample at previously best model configuration
                if len(trial_history) > 0:
                    for best_previous_trial in trial_history:
                        study.enqueue_trial(best_previous_trial)

                study.optimize(objective_1DCNN, n_trials=dict_train_params['N_TRIALS'])

                trial = study.best_trial
                print('------------------------------------------------')
                print('--------------- Optimisation results -----------')
                print('------------------------------------------------')
                print("Number of finished trials: ", len(study.trials))
                print(f"\n           Best trial ({trial.number})        \n")
                print("R2: ", trial.value)
                print("Params: ")
                for key, value in trial.params.items():
                    print("{}: {}".format(key, value))
                trial_history.append(trial.params)

                joblib.dump(study, os.path.join(global_variables.dir_tgt, f'study_{crop_n}_{model_type}.dump'))
                # dumped_study = joblib.load(os.path.join(cst.my_project.meta_dir, 'study_in_memory_storage.dump'))
                # dumped_study.trials_dataframe()
                df = study.trials_dataframe().to_csv(os.path.join(global_variables.dir_tgt, f'study_{crop_n}_{model_type}.csv'))
                # fig = optuna.visualization.plot_slice(study)
                print('------------------------------------------------')

                out_save.save_best_model(global_variables.dir_tgt, f'trial_{trial.number}')

                # Flexible integration for any Python script
                if args.wandb:
                    run_wandb(args, month, trial, fn_asapID2AU, fn_stats90)
    print('Time for this run:')
    print(datetime.datetime.now() - starttime)


def objective_1DCNN(trial):
    global_variables.trial_number = trial.number
    Xt_=Xtk
    Xd = Xt_.shape[1]  # 9, 12, 15, .., 30
    # Suggest values of the hyperparameters using a trial object.
    # nbunits_conv_ = trial.suggest_int('nbunits_conv', 10, 45, step=5)
    nbunits_conv_ = trial.suggest_int('nbunits_conv', 5, 20, step=5)
    kernel_size_ = trial.suggest_categorical('kernel_size', [3, 6])
    pool_size_ = trial.suggest_int('pool_size', 1, Xd//3) #should we fix it at 3, monthly pooling (with max)
    strides_ = pool_size_
    dropout_rate_ = trial.suggest_categorical('dropout_rate', [0, 0.01, 0.1])
    learning_rate_ = trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.01])
    # for the last fully connected layer one cannot request a suggestion seprately as below because nunits_fc_ has no
    # effect when nb_fc_ is 0. This would confuse Optuna
    #nb_fc_ = trial.suggest_categorical('nb_fc', [0, 1])#, 2])
    #nunits_fc_ = trial.suggest_categorical('funits_fc', [16, 32])#16, 64, step=8)
    # Instead we let Optuna to suggest a configuration
    fc_conf = trial.suggest_categorical('fc_conf', [0, 1, 2])
    # Add epochs and batch size as hyper
    # 'N_EPOCHS': 200, # 100, #70,
    n_epochs_ = trial.suggest_int('n_epochs', 30, 210, step=20)
    # 'BATCH_SIZE': 128, #128
    batch_size_ = trial.suggest_categorical('batch_size', [32, 64, 128])

    if fc_conf == 0:
        nb_fc_ = 0
        nunits_fc_ = 0
    elif fc_conf == 1:
        nb_fc_ = 1
        nunits_fc_ = 16
    elif fc_conf == 2:
        nb_fc_ = 1
        nunits_fc_ = 24
    #activation_ = trial.suggest_categorical('activation', ['relu', 'sigmoid'])

    if False: #TODO remove
        nbunits_conv_ = 10
        kernel_size_ = 4
        pool_size_ = 3
        strides_ = pool_size_ #trial.suggest_int('strides', 2, 5)
        dropout_rate_ = 0.01
        nb_fc_ = 0
        nunits_fc_ = 16

    if model_type == '1DCNN_SISO':
        model = Archi_1DCNN('SISO', Xt_,
                                 nbunits_conv=nbunits_conv_,
                                 kernel_size=kernel_size_,
                                 strides=strides_,
                                 pool_size=pool_size_,
                                 dropout_rate=dropout_rate_,
                                 nb_fc=nb_fc_,
                                 nunits_fc=nunits_fc_,
                                 activation='sigmoid',
                                 verbose=False)

    elif model_type == '1DCNN_MISO':
        model = Archi_1DCNN('MISO', Xt_,
                                 Xv= region_ohe,
                                 nbunits_conv=nbunits_conv_,
                                 kernel_size=kernel_size_,
                                 strides=strides_,
                                 pool_size=pool_size_,
                                 dropout_rate=dropout_rate_,
                                 nb_fc=nb_fc_,
                                 nunits_fc=nunits_fc_,
                                 activation='sigmoid',
                                 verbose=False)
    elif model_type == 'simple':
        model = Archi_simple(Xt_,
                             nbunits_conv=nbunits_conv_,
                             kernel_size=kernel_size_,
                             strides=strides_,
                             pool_size=pool_size_,
                             dropout_rate=dropout_rate_,
                             nb_fc=nb_fc_,
                             nunits_fc=nunits_fc_,
                             activation='sigmoid',
                             verbose=False)
        hpsString = 'simple'

    if model_type != 'simple':
        print('Model hypars being tested')
        n_dense_before_output = (len(model.layers) - 1 - 14 - 1) / 2
        hp_dic = {'lr': learning_rate_,
                  'cn_fc4Xv_units': model.layers[1].get_config()['filters'],
                  'cn kernel_size': model.layers[1].get_config()['kernel_size'],
                  'cn drop out rate': model.layers[4].get_config()['rate'],
                  'AveragePooling2D pool_size': model.layers[5].get_config()['pool_size'],
                  'AveragePooling2D strides': model.layers[5].get_config()['strides'],
                  'n FC layers before output (nb_fc)': int(n_dense_before_output),
                  'n_epochs': n_epochs_,
                  'batch_size_': batch_size_
                  }
        dorWithoutDot = str(hp_dic["cn drop out rate"]).replace('.', '-')
        hpsString = f'cnu{hp_dic["cn_fc4Xv_units"]}k{hp_dic["cn kernel_size"][0]}d{dorWithoutDot}' \
                    f'p2Dsz_st{hp_dic["AveragePooling2D pool_size"][0]}_{hp_dic["AveragePooling2D strides"][0]}epo{n_epochs_}batch{batch_size_}'

        for i in range(int(n_dense_before_output)):
            if i == 0:
                hpsString = hpsString + 'dns' + str(model.layers[15 + i * 2].get_config()['units'])
            else:
                hpsString = hpsString + '-' + str(model.layers[15 + i * 2].get_config()['units'])
    print(hpsString)

    # Define output filenames
    fn_fig_val = global_variables.dir_tgt / f'trial_{trial.number}_{hpsString}_val.png'
    fn_fig_test = global_variables.dir_tgt / f'trial_{trial.number}_{hpsString}_test.png'
    fn_cv_test = global_variables.dir_tgt / f'trial_{trial.number}_{hpsString}_test.csv'
    fn_report = global_variables.dir_tgt / f'AAA_report.csv'
    out_model_file = global_variables.dir_tgt / f'{out_model.split(".h5")[0]}_{crop_n}.h5'

    rmses_val, r2s_val, rmses_test, r2s_test = [], [], [], []
    df_val, df_test, df_details, df_bestEpoch = None, None, None, None
    global_variables.init_weights = None


    sampleTerciles = True
    nPerTercile = 2

    global_variables.outer_test_loop = 0
    for test_i in np.unique(groups):
        global_variables.test_group = test_i
        global_variables.inner_cv_loop = 0
        # once the test is excluded, all the others are train and val
        train_val_i = [x for x in np.unique(groups) if x != test_i]
        #TC
        # Xt_test, Xv_test, y_test = readingsits1D.subset_data(Xt, region_ohe, y, groups == test_i)
        # Xt_test = readingsits1D.reshape_data(Xt_test, N_CHANNELS)
        subset_bool = groups == test_i
        Xt_test, Xv_test, y_test = Xt_[subset_bool, :, :], region_ohe[subset_bool, :], y[subset_bool, :]
        # a validation loop on all 16 years of val is too long. We reduce to nPerTercile*3,
        # taking 2 from each tercile of the yield data points
        # Here we assign a tercile to each year. As I have several admin units, I have first to compute avg yield by year
        if sampleTerciles:
            subset_bool = groups > 0 # take all
            Xt_0, Xv_0, y_0 = Xt_[subset_bool, :, :], region_ohe[subset_bool, :], y[subset_bool, :]
            #Xt_0, Xv_0, y_0 = readingsits1D.subset_data(Xt, region_ohe, y, groups > 0)  # take all
            df = pd.DataFrame({'group': groups, 'y': y_0[:, crop_n]})
            df = df[df['group'] != test_i]
            df_avg_by_year = df.groupby('group', as_index=False)['y'].mean()
            q33 = df_avg_by_year['y'].quantile(0.33)
            q66 = df_avg_by_year['y'].quantile(0.66)
            ter1_groups = df_avg_by_year[df_avg_by_year['y'] < q33]['group'].values
            ter2_groups = df_avg_by_year[(df_avg_by_year['y'] >= q33) & (df_avg_by_year['y'] < q66)]['group'].values
            ter3_groups = df_avg_by_year[df_avg_by_year['y'] >= q66]['group'].values
            vals_i = random.sample(ter1_groups.tolist(), nPerTercile)
            vals_i.extend(random.sample(ter2_groups.tolist(), nPerTercile))
            vals_i.extend(random.sample(ter3_groups.tolist(), nPerTercile))
        else:
            vals_i = train_val_i

        for val_i in vals_i: #leave one out for hyper setting (in a way)
            global_variables.val_group = val_i
            # once the val is left out, all the others are train
            train_i = [x for x in train_val_i if x != val_i]
            subset_bool =[x in train_i for x in groups]
            Xt_train, Xv_train, y_train = Xt_[subset_bool, :, :], region_ohe[subset_bool, :], y[subset_bool, :]
            subset_bool = groups == val_i
            Xt_val, Xv_val, y_val = Xt_[subset_bool, :, :], region_ohe[subset_bool, :], y[subset_bool, :]
            #FC
            #Xt_train, Xv_train, y_train = readingsits1D.subset_data(Xt, region_ohe, y, [x in train_i for x in groups])
            #Xt_val, Xv_val, y_val = readingsits1D.subset_data(Xt, region_ohe, y, groups == val_i)
            # ---- Reshaping data necessary
            #Xt_train = readingsits1D.reshape_data(Xt_train, N_CHANNELS)
            #Xt_val = readingsits1D.reshape_data(Xt_val, N_CHANNELS)

            # ---- Normalizing the data per band
            min_per_t, max_per_t = readingsits1D.computingMinMax(Xt_train, per=0)
            Xt_train = readingsits1D.normalizingData(Xt_train, min_per_t, max_per_t)
            Xt_val = readingsits1D.normalizingData(Xt_val, min_per_t, max_per_t)
#            Xt_test = readingsits1D.normalizingData(Xt_test, min_per_t, max_per_t)

            # Normalise ys
            transformer_y = MinMaxScaler().fit(y_train[:, [crop_n]])
            ys_train = transformer_y.transform(y_train[:, [crop_n]])
            ys_val = transformer_y.transform(y_val[:, [crop_n]])
            #ys_test = transformer_y.transform(y_test[:, [crop_n]])

            # TODO remove only one sample, should be learned by hart
            if False:
                idx = range(2)
                Xt_train = Xt_train[idx, :, :].reshape(len(idx), -1, 4)
                Xt_val = Xt_val[idx, :, :].reshape(len(idx), -1, 4)
                ys_train = ys_train[idx,:].reshape(len(idx),-1)
                ys_val = ys_val[idx, :].reshape(len(idx), -1)
                Xv_train = Xv_train[idx,:].reshape(len(idx),-1)
                Xv_val = Xv_val[idx,:].reshape(len(idx),-1)
            # Run the model on train and val data
            if model_type == '1DCNN_SISO':
                model, y_val_preds, bestEpoch = cv_Model(model, {'ts_input': Xt_train}, ys_train,
                                              {'ts_input': Xt_val}, ys_val,
                                              out_model_file,  n_epochs=n_epochs_, batch_size=batch_size_,
                                              learning_rate=hp_dic['lr'], beta_1=dict_train_params['beta_1'],
                                              beta_2=dict_train_params['beta_2'], decay=dict_train_params['decay'])
                X_test = {'ts_input': Xt_test}
            elif model_type == '1DCNN_MISO':
                model, y_val_preds, bestEpoch = cv_Model(model, {'ts_input': Xt_train, 'v_input': Xv_train}, ys_train,
                                              {'ts_input': Xt_val, 'v_input': Xv_val}, ys_val,
                                              out_model_file,  n_epochs=n_epochs_, batch_size=batch_size_,
                                              learning_rate=hp_dic['lr'], beta_1=dict_train_params['beta_1'],
                                              beta_2=dict_train_params['beta_2'], decay=dict_train_params['decay'])
                X_test = {'ts_input': Xt_test, 'v_input': Xv_test}
            elif model_type == 'simple':
                model, y_val_preds, bestEpoch = cv_Model(model, {'ts_input': Xt_train}, ys_train,
                                              {'ts_input': Xt_val}, ys_val,
                                              out_model_file,  n_epochs=n_epochs_, batch_size=batch_size_,
                                              learning_rate=hp_dic['lr'], beta_1=dict_train_params['beta_1'],
                                              beta_2=dict_train_params['beta_2'], decay=dict_train_params['decay'])
                X_test = {'ts_input': Xt_test}

            y_val_preds = transformer_y.inverse_transform(y_val_preds)
            out_val = np.concatenate([y_val[:, [crop_n]], y_val_preds], axis=1)

            if df_val is None:
                df_val = out_val
            else:
                df_val = np.concatenate([df_val, out_val], axis=0)

            if df_bestEpoch is None:
                df_bestEpoch = np.array(bestEpoch)
            else:
                df_bestEpoch = np.append(df_bestEpoch, bestEpoch)

            rmse_val = mean_squared_error(y_val[:, [crop_n]], y_val_preds, squared=False, multioutput='raw_values')
            r2_val = r2_score(y_val[:, [crop_n]], y_val_preds)
            rmses_val.append(rmse_val)
            r2s_val.append(r2_val)

            # Update counter
            global_variables.inner_cv_loop += 1
            #inner_loop += 1

        # CV finished
        global_variables.outer_test_loop += 1

        print(
            f'Outer loop {global_variables.outer_test_loop} - with {global_variables.inner_cv_loop} inner loop, testing n epochs: {n_epochs_ }, best at: {df_bestEpoch}')
        df_bestEpoch = None
        # ---- Optuna pruning
        if dict_train_params['optuna_metric'] == 'rmse':
            varOptuna = np.mean(rmses_val)
        elif dict_train_params['optuna_metric'] == 'r2':
            varOptuna = np.mean(r2s_val)

        trial.report(varOptuna, global_variables.outer_test_loop)  # report mse
        if trial.should_prune():  # let optuna decide whether to prune dir_tgt / f'trial_{trial.number}_{hpsString}
            # save configuration and performances in a file
            df_report = pd.DataFrame([[trial.number,'@outer_loop'+str(global_variables.outer_test_loop), hp_dic['lr'], np.mean(rmses_val), np.mean(r2s_val), np.mean(rmses_test), np.mean(r2s_test),
                                             nbunits_conv_, kernel_size_, pool_size_, strides_, dropout_rate_, nb_fc_, nunits_fc_, n_epochs_, batch_size_]],
                                     columns=['Trial', 'Pruned', 'lr', 'av_rmse_val', 'av_r2_val', 'av_rmse_test', 'av_r2_test',
                                              'nbunits_conv', 'kernel_size', 'pool_size', 'strides', 'dropout_rate', 'n_fc', 'nunits_fc', 'n_epochs', 'batch_size'])
            if os.path.exists(fn_report):
                df_report.to_csv(fn_report, mode='a', header=False)
            else:
                df_report.to_csv(fn_report)
            raise optuna.exceptions.TrialPruned()


        # Fit the model on training and validation data
        subset_bool = [x in train_val_i for x in groups]
        Xt_train, Xv_train, y_train = Xt_[subset_bool, :, :], region_ohe[subset_bool, :], y[subset_bool, :]
        #TC
        #Xt_train, Xv_train, y_train = readingsits1D.subset_data(Xt, region_ohe, y,                                                           [x in train_val_i for x in groups])
        # ---- Reshaping data necessary
        #Xt_train = readingsits1D.reshape_data(Xt_train, N_CHANNELS)
        # ---- Normalizing the data per band
        min_per_t, max_per_t = readingsits1D.computingMinMax(Xt_train, per=0)
        Xt_train = readingsits1D.normalizingData(Xt_train, min_per_t, max_per_t)
        Xt_test = readingsits1D.normalizingData(Xt_test, min_per_t, max_per_t)
        # Normalise ys
        transformer_y = MinMaxScaler().fit(y_train[:, [crop_n]])
        ys_train = transformer_y.transform(y_train[:, [crop_n]])
        if model_type == '1DCNN_SISO':
            model, y_val_preds, bestEpoch = cv_Model(model, {'ts_input': Xt_train}, ys_train,
                                                     {'ts_input': None}, None,
                                                     out_model_file,  nEpochs4FinalFit=n_epochs_,n_epochs=n_epochs_, batch_size=batch_size_,
                                                     learning_rate=hp_dic['lr'],
                                                     beta_1=dict_train_params['beta_1'],
                                                     beta_2=dict_train_params['beta_2'],
                                                     decay=dict_train_params['decay'])
            X_test = {'ts_input': Xt_test}
        elif model_type == '1DCNN_MISO':
            model, y_val_preds, bestEpoch = cv_Model(model, {'ts_input': Xt_train, 'v_input': Xv_train}, ys_train,
                                                     {'ts_input': None, 'v_input': None}, None,
                                                     out_model_file, nEpochs4FinalFit=n_epochs_, n_epochs=n_epochs_, batch_size=batch_size_,
                                                     learning_rate=hp_dic['lr'],
                                                     beta_1=dict_train_params['beta_1'],
                                                     beta_2=dict_train_params['beta_2'],
                                                     decay=dict_train_params['decay'])
            X_test = {'ts_input': Xt_test, 'v_input': Xv_test}
        elif model_type == 'simple':
            model, y_val_preds, bestEpoch = cv_Model(model, {'ts_input': Xt_train}, ys_train,
                                                     {'ts_input': None}, None,
                                                     out_model_file, nEpochs4FinalFit=selectedEpochs,
                                                     n_epochs=dict_train_params['N_EPOCHS'],
                                                     batch_size=dict_train_params['BATCH_SIZE'],
                                                     learning_rate=hp_dic['lr'],
                                                     beta_1=dict_train_params['beta_1'],
                                                     beta_2=dict_train_params['beta_2'],
                                                     decay=dict_train_params['decay'])
            X_test = {'ts_input': Xt_test}
        # Now make prediction using all data in training and number of epochs from df_bestEpoch
        y_test_preds = model.predict(x=X_test)
        y_test_preds = transformer_y.inverse_transform(y_test_preds)

        out_test = np.concatenate([y_test[:, [crop_n]], y_test_preds], axis=1)
        out_details = np.expand_dims(region_id[groups == test_i].T, axis=1)
        if not isinstance(df_details, np. ndarray): #df_details == None:
            df_details = np.concatenate([out_details, (np.ones_like(out_details) * test_i)], axis=1)
            df_test = out_test
        else:
            df_details = np.concatenate(
                [df_details, np.concatenate([out_details, (np.ones_like(out_details) * test_i)], axis=1)], axis=0)
            df_test = np.concatenate([df_test, out_test], axis=0)
        rmse_test = mean_squared_error(y_test[:, [crop_n]], y_test_preds, squared=False, multioutput='raw_values')
        r2_test = r2_score(y_test[:, [crop_n]], y_test_preds)
        rmses_test.append(rmse_test)
        r2s_test.append(r2_test)

    #test loop ended
    # Compute by cv folder average statistics (all excluding r2 test wich is compute in plotting)
    av_rmse_val = np.mean(rmses_val)
    av_r2_val = np.mean(r2s_val)
    av_rmse_test = np.mean(rmses_test)

    # out_plot.plot_val_test_predictions(df_val, df_test, av_rmse_val, r2s_val, av_rmse_test, r2s_test, xlabels, ylabels,
    #                           filename_val=fn_fig_val, filename_test=fn_fig_test)
    out_plot.plot_val_test_predictions_with_details(df_val, df_test, av_rmse_val, r2s_val, av_rmse_test, r2s_test, xlabels, ylabels, df_details,
                                       filename_val=fn_fig_val, filename_test=fn_fig_test)
    # Save CV results
    df_out = np.concatenate([df_details, df_test], axis=1)
    pd.DataFrame(df_out, columns=['ASAP1_ID', 'Year', 'Observed', 'Predicted']).to_csv(fn_cv_test, index=False)

    # save configuration and performances in a file
    # df_report = pd.DataFrame(data=[['no', hp_dic['lr'], av_rmse_val, av_r2_val, av_rmse_test, np.mean(r2s_test),
    #                           nbunits_conv_, kernel_size_, pool_size_, strides_, dropout_rate_, nb_fc_, nunits_fc_]],
    #                          columns=['Pruned','lr','av_rmse_val', 'av_r2_val', 'av_rmse_test', 'av_r2_test',
    #                                   'nbunits_conv', 'kernel_size', 'pool_size', 'strides', 'dropout_rate', 'n_fc',
    #                                   'nunits_fc'])
    df_report = pd.DataFrame([[trial.number, 'no', hp_dic['lr'],
                               av_rmse_val, av_r2_val, av_rmse_test, np.mean(r2s_test),
                               nbunits_conv_, kernel_size_, pool_size_, strides_, dropout_rate_, nb_fc_, nunits_fc_, n_epochs_, batch_size_]],
                             columns=['Trial', 'Pruned', 'lr', 'av_rmse_val', 'av_r2_val', 'av_rmse_test', 'av_r2_test',
                                      'nbunits_conv', 'kernel_size', 'pool_size', 'strides', 'dropout_rate', 'n_fc',
                                      'nunits_fc', 'n_epochs', 'batch_size'])
    if os.path.exists(fn_report):
        df_report.to_csv(fn_report, mode='a', header=False)
    else:
        df_report.to_csv(fn_report)

    if dict_train_params['optuna_metric'] == 'rmse':
        return av_rmse_val
    elif dict_train_params['optuna_metric'] == 'r2':
        return av_r2_val




def run_wandb(args, month, trial, fn_asapID2AU, fn_stats90):
    # 1. Start a W&B run
    wandb.init(project=cst.wandb_project, entity=cst.wandb_entity, reinit=True,
               group=f'{args.target} - {crop_n} - {month}', config=trial.params,
               name=f'{args.target}-{model_type}-{crop_n}-{month}',
               notes=f'Performance of a 1D CNN model for {args.target} forecasting in Algeria for'
                     f'crop ID {crop_n}.')
    # 2. Save model inputs and hyperparameters
    wandb.config.update({'model_type': model_type,
                         'crop_n': crop_n,
                         'month': month,
                         'target': args.target,
                         'n_epochs': dict_train_params['N_EPOCHS'],
                         'batch_size': dict_train_params['BATCH_SIZE'],
                         'n_trials': dict_train_params['N_TRIALS'],
                         })

    # Evaluate best model on test set
    fn_csv_best = [x for x in (global_variables.dir_tgt / 'best_model').glob('*.csv')][0]
    res_i = mod_eval.model_evaluation(fn_csv_best, crop_n, month, model_type, fn_asapID2AU, fn_stats90)
    # 3. Log metrics over time to visualize performance
    wandb.log({"crop_n": crop_n,
               "month": month,
               "R2_p": res_i.R2_p.to_numpy()[0],
               "MAE_p": res_i.MAE_p.to_numpy()[0],
               "rMAE_p": res_i.rMAE_p.to_numpy()[0],
               "ME_p": res_i.ME_p.to_numpy()[0],
               "RMSE_p": res_i.RMSE_p.to_numpy()[0],
               "rRMSE_p": res_i.rRMSE_p.to_numpy()[0],
               "Country_R2_p": res_i.Country_R2_p.to_numpy()[0],
               "Country_MAE_p": res_i.Country_MAE_p.to_numpy()[0],
               "Country_ME_p": res_i.Country_ME_p.to_numpy()[0],
               "Country_RMSE_p": res_i.Country_RMSE_p.to_numpy()[0],
               "Country_rRMSE_p": res_i.Country_rRMSE_p.to_numpy()[0],
               "Country_FQ_rRMSE_p": res_i.Country_FQ_rRMSE_p.to_numpy()[0],
               "Country_FQ_RMSE_p": res_i.Country_FQ_RMSE_p.to_numpy()[0]
               })

    wandb.finish()


# -----------------------------------------------------------------------
if __name__ == "__main__":
    main()