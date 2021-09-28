#!/usr/bin/python

import os
import sys
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

random.seed(4)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False

from deeplearning.architecture_features import cv_Model
from deeplearning.architecture_complexity_2D import Archi_2DCNN_MISO, Archi_2DCNN_SISO
from outputfiles import plot as out_plot
from outputfiles import save as out_save
from evaluation import model_evaluation as mod_eval
from sits import readingsits2D
import mysrc.constants as cst
import sits.data_generator as data_generator

# global vars - used in objective_2DCNN
model_type = None
Xt = None
region_ohe = None
dir_tgt = None
groups = None
data_augmentation = None
generator = None
y = None
crop_n = None
n_epochs = None
batch_size = None
region_id = None
xlabels = None
ylabels = None
out_model = None


def main():
    # ---- Define parser
    parser = argparse.ArgumentParser(description='Optimise 2D CNN for yield and area forecasting')
    parser.add_argument('--normalisation', type=str, default='norm', choices=['norm', 'raw'],
                        help='Should input data be normalised histograms?')
    parser.add_argument('--model', type=str, default='2DCNN_SISO',
                        help='Model type: Single input single output (SISO) or Multiple inputs/Single output (MISO)')
    parser.add_argument('--target', type=str, default='yield', choices=['yield', 'area'], help='Target variable')
    parser.add_argument('--Xshift', dest='Xshift', action='store_true', default=False, help='Data aug, shiftX')
    parser.add_argument('--Xnoise', dest='Xnoise', action='store_true', default=False, help='Data aug, noiseX')
    parser.add_argument('--Ynoise', dest='Ynoise', action='store_true', default=False, help='Data aug, noiseY')
    parser.add_argument('--wandb', dest='wandb', action='store_true', default=False, help='Store results on wandb.io')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', default=False,
                        help='Overwrite existing results')
    # parser.add_argument('data augmentation', type=int, default='+', help='an integer for the accumulator')
    args = parser.parse_args()

    # ---- Parameters to set
    n_channels = 4  # -- NDVI, Rad, Rain, Temp
    global n_epochs, batch_size
    n_epochs = 70
    batch_size = 128
    n_trials = 100

    # ---- Get parameters
    global model_type
    model_type = args.model
    if args.wandb:
        print('Wandb log requested')
    da_label = ''
    global data_augmentation
    if args.Xshift or args.Xnoise or args.Ynoise:
        data_augmentation = True
        if args.Xshift == True:
            da_label = da_label + 'Xshift'
        if args.Xnoise == True:
            da_label = da_label + '_Xnoise'
        if args.Ynoise == True:
            da_label = da_label + '_Ynoise'
    else:
        data_augmentation = False

    # ---- Define some paths to data
    fn_indata = cst.my_project.data_dir / f'{cst.target}_full_2d_dataset_raw.pickle'
    print("Input file: ", os.path.basename(str(fn_indata)))

    fn_asapID2AU = cst.root_dir / "raw_data" / "Algeria_REGION_id.csv"
    fn_stats90 = cst.root_dir / "raw_data" / "Algeria_stats90.csv"

    # for input_size in [32, 48, 64]:
    for input_size in [64, 32]:
        # ---- Downloading (always not normalized)
        Xt_full, area_full, region_id_full, groups_full, yld_full = readingsits2D.data_reader(fn_indata)

        # M+ original resizing of Franz using tf.image.resize was bit odd as it uses bilinear interp (filling thus zeros)
        # resize if required (only resize to 32 possible)
        if input_size != 64:
            if input_size == 32:
                Xt_full = Xt_full.reshape(Xt_full.shape[0], -1, 2, Xt_full.shape[-2], Xt_full.shape[-1]).sum(2)
            else:
                print("Resizing request is not available")
                sys.exit()

        if args.normalisation == 'norm':
            max_per_image = np.max(Xt_full, axis=(1, 2), keepdims=True)
            Xt_full = Xt_full / max_per_image
        # M-

        # loop through all crops
        global crop_n
        for crop_n in [1, 2]:  # range(y.shape[1]): TODO: now processing the two missing (0 - Barley, 1 - Durum, 2- Soft)
            # clean trial history for a new crop
            trial_history = []

            # make sure that we do not keep entries with 0 ton/ha yields,
            yields_2_keep = ~(yld_full[:, crop_n] <= 0)
            Xt_nozero = Xt_full[yields_2_keep, :, :, :]
            area = area_full[yields_2_keep, :]
            global region_id, groups
            region_id = region_id_full[yields_2_keep]
            groups = groups_full[yields_2_keep]
            yld = yld_full[yields_2_keep, :]
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
            region_ohe = readingsits2D.add_one_hot(region_id)

            # loop by month
            for month in range(1, cst.n_month_analysis + 1):
                # ---- output files and dirs
                dir_out = cst.my_project.params_dir
                dir_out.mkdir(parents=True, exist_ok=True)
                dir_res = dir_out / f'Archi_{str(model_type)}_{args.target}_{args.normalisation}_{input_size}_{da_label}'
                dir_res.mkdir(parents=True, exist_ok=True)
                global out_model
                out_model = f'archi-{model_type}-{args.target}-{args.normalisation}.h5'
                # crop dirs
                dir_crop = dir_res / f'crop_{crop_n}'
                dir_crop.mkdir(parents=True, exist_ok=True)
                # month dirs
                global dir_tgt
                dir_tgt = dir_crop / f'month_{month}'
                dir_tgt.mkdir(parents=True, exist_ok=True)

                if data_augmentation:
                    # Instantiate a data generator for this crop
                    global generator
                    generator = data_generator.DG(Xt_nozero, region_ohe, y, Xshift=args.Xshift, Xnoise=args.Xnoise,
                                                  Ynoise=args.Ynoise)

                if (len([x for x in dir_tgt.glob('best_model')]) != 0) & (args.overwrite is False):
                    pass
                else:
                    # Clean up directory if incomplete run of if overwrite is True
                    out_save.rm_tree(dir_tgt)
                    # data start in first dek of August (cst.first_month_in__raw_data), index 0
                    # the model uses data from first dek of September (to account for precipitation, field preparation),
                    # cst.first_month_input_local_year, =1, 1*3, index 3
                    # first forecast (month 1) is using up to end of Nov, index 11
                    first = (cst.first_month_input_local_year) * 3
                    last = (cst.first_month_analysis_local_year + month - 1) * 3  # this is 12
                    global Xt
                    Xt = Xt_nozero[:, :, first:last, :]  # this takes 9 elements, from 3 to 11 included

                    print('------------------------------------------------')
                    print('------------------------------------------------')
                    print(f"")
                    print(f'=> noarchi: {model_type} - normalisation: {args.normalisation} - target:'
                          f' {args.target} - crop: {crop_n} - month: {month} =')
                    print(f'Training data have shape: {Xt.shape}')

                    study = optuna.create_study(direction='maximize',
                                                sampler=TPESampler(),
                                                pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=8)
                                                )

                    # Force the sampler to sample at previously best model configuration
                    if len(trial_history) > 0:
                        for best_previous_trial in trial_history:
                            study.enqueue_trial(best_previous_trial)

                    study.optimize(objective_2DCNN, n_trials=n_trials)

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

                    joblib.dump(study, os.path.join(dir_tgt, f'study_{crop_n}_{model_type}.dump'))
                    # dumped_study = joblib.load(os.path.join(cst.my_project.meta_dir, 'study_in_memory_storage.dump'))
                    # dumped_study.trials_dataframe()
                    df = study.trials_dataframe().to_csv(os.path.join(dir_tgt, f'study_{crop_n}_{model_type}.csv'))
                    # fig = optuna.visualization.plot_slice(study)
                    print('------------------------------------------------')

                    out_save.save_best_model(dir_tgt, f'res_{trial.number}')

                    # Flexible integration for any Python script
                    if args.wandb:
                        run_wandb(args, month, input_size, trial, da_label, n_trials, fn_asapID2AU, fn_stats90)


def objective_2DCNN(trial):
    # Suggest values of the hyperparameters using a trial object.
    nbunits_conv_ = trial.suggest_int('nbunits_conv', 8, 48, step=4)
    kernel_size_ = trial.suggest_int('kernel_size', 3, 6)
    strides_ = trial.suggest_int('strides', 1, 6) # MAKE IT POOL SIZE x
    pool_size_ = trial.suggest_int('pool_size', 1, 6) # POOL SIZE Y, and let strides = pool size (//2 on time axis)
    pyramid_bins_ = trial.suggest_int('pyramid_bin', 1, 4)
    pyramid_bins_ = [[k,k] for k in np.arange(1, pyramid_bins_+1)]
    dropout_rate_ = trial.suggest_float('dropout_rate', 0, 0.2, step=0.1)
    nb_fc_ = trial.suggest_categorical('nb_fc', [1, 2, 3]) #as Franz coded afterwards this menas 0, 1, 2 layers
    nunits_fc_ = trial.suggest_int('funits_fc', 16, 64, step=8) #the additional fc layer will have n, n/2, n/4 units
    #activation_ = trial.suggest_categorical('activation', ['relu', 'sigmoid'])

    if model_type == '2DCNN_SISO':
        model = Archi_2DCNN_SISO(Xt,
                                 nbunits_conv=nbunits_conv_,
                                 kernel_size=kernel_size_,
                                 strides=strides_,
                                 pool_size=pool_size_,
                                 pyramid_bins=pyramid_bins_,
                                 dropout_rate=dropout_rate_,
                                 nb_fc=nb_fc_,
                                 nunits_fc=nunits_fc_,
                                 activation='sigmoid',
                                 verbose=False)

    elif model_type == '2DCNN_MISO':
        model = Archi_2DCNN_MISO(Xt,
                                 region_ohe,
                                 nbunits_conv=nbunits_conv_,
                                 kernel_size=kernel_size_,
                                 strides=strides_,
                                 pool_size=pool_size_,
                                 pyramid_bins=pyramid_bins_,
                                 dropout_rate=dropout_rate_,
                                 nb_fc=nb_fc_,
                                 nunits_fc=nunits_fc_,
                                 activation='sigmoid',
                                 verbose=False)
    print('Model hypars being tested')
    n_dense_before_output = (len(model.layers) - 1 - 14 - 1) / 2
    hp_dic = {'cn_fc4Xv_units': str(model.layers[1].get_config()['filters']),
              'cn kernel_size': str(model.layers[1].get_config()['kernel_size']),
              #'cn strides (fixed)': str(model.layers[1].get_config()['strides']),
              'cn drop out rate:': str(model.layers[4].get_config()['rate']),
              'AveragePooling2D pool_size': str(model.layers[5].get_config()['pool_size']),
              'AveragePooling2D strides': str(model.layers[5].get_config()['strides']),
              'SpatialPyramidPooling2D bins': str(model.layers[10].get_config()['bins']),
              'n FC layers before output (nb_fc)': str(int(n_dense_before_output))
              }
    for i in range(int(n_dense_before_output)):
        hp_dic[str(i) + ' ' + 'fc_units'] = str(model.layers[15 + i * 2].get_config()['units'])
        hp_dic[str(i) + ' ' + 'drop out rate'] = str(model.layers[16 + i * 2].get_config()['rate'])
    print(hp_dic.values())
    # Define output filenames
    fn_fig_val = dir_tgt / f'{out_model.split(".h5")[0]}_res_{trial.number}_val.png'
    fn_fig_test = dir_tgt / f'{out_model.split(".h5")[0]}_res_{trial.number}_test.png'
    fn_cv_test = dir_tgt / f'{out_model.split(".h5")[0]}_res_{trial.number}_test.csv'
    out_model_file = dir_tgt / f'{out_model.split(".h5")[0]}_{crop_n}.h5'

    mses_val, r2s_val, mses_test, r2s_test = [], [], [], []
    df_val, df_test, df_details = None, None, None
    cv_i = 0
    for test_i in np.unique(groups):
        val_i = random.choice([x for x in np.unique(groups) if x != test_i])
        train_i = [x for x in np.unique(groups) if x != val_i and x != test_i]

        # Create train, val and test sets
        train_indices = [x in train_i for x in groups]
        Xt_train, Xv_train, y_train = readingsits2D.subset_data(Xt, region_ohe, y, train_indices)
        # training data augmentation
        if data_augmentation:
            Xt_train, Xv_train, y_train = generator.generate(Xt_train.shape[2], train_indices)

        Xt_val, Xv_val, y_val = readingsits2D.subset_data(Xt, region_ohe, y, groups == val_i)
        Xt_test, Xv_test, y_test = readingsits2D.subset_data(Xt, region_ohe, y, groups == test_i)

        # removed, this is not the right way, resampling has now been made upfront in main
        #X_train = tf.image.resize(Xt_train, [input_size, input_size]).numpy()
        #Xt_val = tf.image.resize(Xt_train, [input_size, input_size]).numpy()
        #Xt_test = tf.image.resize(Xt_train, [input_size, input_size]).numpy()

        # If images are already normalised per region, the following has no effect
        # if not this is a minmax scaling based on the training set.
        # WARNING: if data are normalized by region (and not by image), the following normalisation would have an effect
        min_per_t, max_per_t = readingsits2D.computingMinMax(Xt_train, per=0)
        # Normalise training set
        Xt_train = readingsits2D.normalizingData(Xt_train, min_per_t, max_per_t)
        # print(f'Shape training data: {Xt_train.shape}')
        # Normalise validation set
        Xt_val = readingsits2D.normalizingData(Xt_val, min_per_t, max_per_t)
        # Normalise test set
        Xt_test = readingsits2D.normalizingData(Xt_test, min_per_t, max_per_t)

        # Normalise ys
        transformer_y = MinMaxScaler().fit(y_train[:, [crop_n]])
        ys_train = transformer_y.transform(y_train[:, [crop_n]])
        ys_val = transformer_y.transform(y_val[:, [crop_n]])
        ys_test = transformer_y.transform(y_test[:, [crop_n]])

        # We compile our model with a sampled learning rate.
        if model_type == '2DCNN_SISO':
            model, y_val_preds = cv_Model(model, {'ts_input': Xt_train}, ys_train,
                                          {'ts_input': Xt_val}, ys_val,
                                          out_model_file, n_epochs=n_epochs, batch_size=batch_size)
            X_test = {'ts_input': Xt_test}
        elif model_type == '2DCNN_MISO':
            model, y_val_preds = cv_Model(model, {'ts_input': Xt_train, 'v_input': Xv_train}, ys_train,
                                          {'ts_input': Xt_val, 'v_input': Xv_val}, ys_val,
                                          out_model_file, n_epochs=n_epochs, batch_size=batch_size)
            X_test = {'ts_input': Xt_test, 'v_input': Xv_test}

        y_val_preds = transformer_y.inverse_transform(y_val_preds)
        out_val = np.concatenate([y_val[:, [crop_n]], y_val_preds], axis=1)

        y_test_preds = model.predict(x=X_test)
        y_test_preds = transformer_y.inverse_transform(y_test_preds)
        out_test = np.concatenate([y_test[:, [crop_n]], y_test_preds], axis=1)
        out_details = np.expand_dims(region_id[groups == test_i].T, axis=1)
        if df_val is None:
            df_val = out_val
            df_test = out_test
            df_details = np.concatenate([out_details, (np.ones_like(out_details) * test_i)], axis=1)
        else:
            df_val = np.concatenate([df_val, out_val], axis=0)
            df_test = np.concatenate([df_test, out_test], axis=0)
            df_details = np.concatenate([df_details,
                                         np.concatenate([out_details, (np.ones_like(out_details) * test_i)], axis=1)],
                                        axis=0)

        # It happens that the trial results in  y_val_preds being nan because model fit failed with given optuna params and data
        # To avoid rasin nan errors in computation of stats below we handle this here
        if np.isnan(y_val_preds).any():
            mses_val.append(np.nan)
            r2s_val.append(np.nan)
            mse_test = np.nan
            r2_test = np.nan
            mses_test.append(mse_test)
            r2s_test.append(r2_test)
            # ---- Optuna pruning
            trial.report(np.nan, cv_i)  # report mse
        else:
            mse_val = mean_squared_error(y_val[:, [crop_n]], y_val_preds, squared=False, multioutput='raw_values')
            r2_val = r2_score(y_val[:, [crop_n]], y_val_preds)
            mses_val.append(mse_val)
            r2s_val.append(r2_val)
            mse_test = mean_squared_error(y_test[:, [crop_n]], y_test_preds, squared=False, multioutput='raw_values')
            r2_test = r2_score(y_test[:, [crop_n]], y_test_preds)
            mses_test.append(mse_test)
            r2s_test.append(r2_test)
            # ---- Optuna pruning
            trial.report(np.mean(r2s_val), cv_i)  # report mse


        if trial.should_prune():  # let optuna decide whether to prune
            raise optuna.exceptions.TrialPruned()

        # Update counter
        cv_i += 1

    if ~np.isnan(y_val_preds).any():
        av_rmse_val = np.mean(mses_val)
        av_r2_val = np.mean(r2s_val)
        av_rmse_test = np.mean(mses_test)
        out_plot.plot_val_test_predictions(df_val, df_test, av_rmse_val, r2s_val, av_rmse_test, r2s_test, xlabels, ylabels,
                              filename_val=fn_fig_val, filename_test=fn_fig_test)

    # Save CV results
    df_out = np.concatenate([df_details, df_test], axis=1)
    pd.DataFrame(df_out, columns=['ASAP1_ID', 'Year', 'Observed', 'Predicted']).to_csv(fn_cv_test, index=False)

    return av_r2_val


def run_wandb(args, month, input_size, trial, da_label, n_trials, fn_asapID2AU, fn_stats90):
    # 1. Start a W&B run
    wandb.init(project=cst.wandb_project, entity=cst.wandb_entity, reinit=True,
               group=f'{args.target}C{crop_n}M{month}SZ{input_size}', config=trial.params,
               name=f'{args.target}-{model_type}-C{crop_n}-M{month}-{args.normalisation}-{da_label}',
               notes=f'Performance of a 2D CNN model for {args.target} forecasting in Algeria for'
                     f'crop ID {crop_n}.')

    # 2. Save model inputs and hyperparameters
    wandb.config.update({'model_type': model_type,
                         'crop_n': crop_n,
                         'month': month,
                         'norm': args.normalisation,
                         'target': args.target,
                         'n_epochs': n_epochs,
                         'batch_size': batch_size,
                         'n_trials': n_trials,
                         'input_size': input_size
                         })

    # Evaluate best model on test set
    fn_csv_best = [x for x in (dir_tgt / 'best_model').glob('*.csv')][0]
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