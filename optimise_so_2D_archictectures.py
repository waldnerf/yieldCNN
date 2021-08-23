#!/usr/bin/python

import os, sys
import argparse
import random
import shutil
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import optuna
from optuna.samplers import TPESampler
import joblib
import random
import wandb
import tensorflow as tf

random.seed(4)

# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')

# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False

from deeplearning.architecture_complexity_2d import *
from outputfiles.plot import *
from outputfiles.save import *
from model_evaluation import *
from outputfiles.evaluation import *
from sits.readingsits2D import *
import mysrc.constants as cst


def objective_2DCNN(trial):
    # Suggest values of the hyperparameters using a trial object.
    nbunits_conv_ = trial.suggest_int('nbunits_conv', 8, 48, step=4)
    kernel_size_ = trial.suggest_int('kernel_size', 3, 6)
    strides_ = trial.suggest_int('strides', 1, 6) # MAKE IT POOL SIZE x
    pool_size_ = trial.suggest_int('pool_size', 1, 6) # POOL SIZE Y, and let strides = pool size (//2 on time axis)
    pyramid_bins_ = trial.suggest_int('pyramid_bin', 1, 4)
    pyramid_bins_ = [[k,k] for k in np.arange(1, pyramid_bins_+1)]
    dropout_rate_ = trial.suggest_float('dropout_rate', 0, 0.2, step=0.1)
    nb_fc_ = trial.suggest_categorical('nb_fc', [1, 2, 3])
    nunits_fc_ = trial.suggest_int('funits_fc', 16, 64, step=8)
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

        Xt_train, Xv_train, y_train = subset_data(Xt, region_ohe, y, [x in train_i for x in groups])
        Xt_val, Xv_val, y_val = subset_data(Xt, region_ohe, y, groups == val_i)
        Xt_test, Xv_test, y_test = subset_data(Xt, region_ohe, y, groups == test_i)

        # If images are already normalised per region, the following has no effect
        # if not this is a minmax scaling based on the training set.
        min_per_t, max_per_t = computingMinMax(Xt_train, per=0)
        # Normalise training set
        Xt_train = normalizingData(Xt_train, min_per_t, max_per_t)
        # print(f'Shape training data: {Xt_train.shape}')
        # Normalise validation set
        Xt_val = normalizingData(Xt_val, min_per_t, max_per_t)
        # Normalise test set
        Xt_test = normalizingData(Xt_test, min_per_t, max_per_t)

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

    av_rmse_val = np.mean(mses_val)
    av_r2_val = np.mean(r2s_val)
    av_rmse_test = np.mean(mses_test)

    plot_val_test_predictions(df_val, df_test, av_rmse_val, r2s_val, av_rmse_test, r2s_test, xlabels, ylabels,
                              filename_val=fn_fig_val, filename_test=fn_fig_test)

    # Save CV results
    df_out = np.concatenate([df_details, df_test], axis=1)
    pd.DataFrame(df_out, columns=['ASAP1_ID', 'Year', 'Observed', 'Predicted']).to_csv(fn_cv_test, index=False)

    return av_r2_val


# -----------------------------------------------------------------------
if __name__ == "__main__":
    # ---- Define parser
    parser = argparse.ArgumentParser(description='Optimise 2D CNN for yield and area forecasting')
    parser.add_argument('--normalisation', type=str, default='norm', choices=['norm', 'raw'], help='Should input data be normalised histograms?')
    parser.add_argument('--model', type=str, default='2DCNN_SISO',
                        help='Model type: Single input single output (SISO) or Multiple inputs/Single output (MISO)')
    parser.add_argument('--target', type=str, default='yield', choices=['yield', 'area'], help='Target variable')
    parser.add_argument('--wandb', type=bool, default=True, help='Store results on wandb.io')
    parser.add_argument('--overwrite', type=bool, default=False, help='Overwrite existing results')
    # parser.add_argument('data augmentation', type=int, default='+', help='an integer for the accumulator')
    args = parser.parse_args()

    # ---- Parameters to set
    n_channels = 4  # -- NDVI, Rad, Rain, Temp
    n_epochs = 70
    batch_size = 128
    n_trials = 100

    # ---- Get parameters
    model_type = args.model
    target_var = args.target
    wandb_log = args.wandb
    overwrite = args.overwrite


    # ---- Define some paths to data
    if args.normalisation == 'norm':
        fn_indata = cst.my_project.data_dir / f'{cst.target}_full_2d_dataset_norm.pickle'
        hist_norm = 'norm'
    else:
        fn_indata = cst.my_project.data_dir / f'{cst.target}_full_2d_dataset_raw.pickle'
        hist_norm = 'raw'
    print("Input file: ", os.path.basename(str(fn_indata)))

    fn_asapID2AU = cst.root_dir / "raw_data" / "Algeria_REGION_id.csv"
    fn_stats90 = cst.root_dir / "raw_data" / "Algeria_stats90.csv"

    for input_size in [64, 32]:
        # ---- output files
        dir_out = cst.my_project.params_dir
        dir_out.mkdir(parents=True, exist_ok=True)
        dir_res = dir_out / f'Archi_{str(model_type)}_{target_var}_{hist_norm}_{input_size}'
        dir_res.mkdir(parents=True, exist_ok=True)
        out_model = f'archi-{model_type}-{target_var}-{hist_norm}.h5'

        # ---- Downloading
        Xt_full, area_full, region_id_full, groups_full, yld_full = data_reader(fn_indata)

        trial_history = []
        # loop through all crops
        for crop_n in [1]:  # range(y.shape[1]): TODO: only process durum wheat!
            dir_crop = dir_res / f'crop_{crop_n}'
            dir_crop.mkdir(parents=True, exist_ok=True)

            # make sure that we do not keep entries with 0 ton/ha yields,
            yields_2_keep = ~(yld_full[:, crop_n] <= 0)
            Xt_nozero = Xt_full[yields_2_keep, :, :, :]
            area = area_full[yields_2_keep, :]
            region_id = region_id_full[yields_2_keep]
            groups = groups_full[yields_2_keep]
            yld = yld_full[yields_2_keep, :]
            # ---- Format target variable
            if target_var == 'yield':
                y = yld
                xlabels = 'Predictions (t/ha)'
                ylabels = 'Observations (t/ha)'
            elif target_var == 'area':
                y = area
                xlabels = 'Predictions (%)'
                ylabels = 'Observations (%)'

            # ---- Convert region to one hot
            region_ohe = add_one_hot(region_id)

            # loop by month
            for month in range(1, 9):
                dir_tgt = dir_crop / f'month_{month}'
                dir_tgt.mkdir(parents=True, exist_ok=True)
                if (len([x for x in dir_tgt.glob('best_model')]) != 0) & (overwrite is False):
                    pass
                else:
                    # Clean up directory if incomplete run of if overwrite is True
                    rm_tree(dir_tgt)

                    # as we said we start in Sep now so first forecast (month 1) is using up to end of Nov
                    Xt = Xt_nozero[:, :, 3:(3 + (2 + month) * 3), :]
                    if input_size == 32:
                        max_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 1), strides=(2, 1), padding='same')
                        Xt = max_pool_2d(Xt).numpy()

                    print('------------------------------------------------')
                    print('------------------------------------------------')
                    print(f"")
                    print(f'=> noarchi: {model_type} - normalisation: {hist_norm} - target:'
                          f' {target_var} - crop: {crop_n} - month: {month} =')
                    print(f'Training data have shape: {Xt.shape}')

                    study = optuna.create_study(direction='maximize',
                                                sampler=TPESampler(),
                                                pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=8)
                                                )

                    # Force the sample to sample at previously best model configuration
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

                    save_best_model(dir_tgt, f'res_{trial.number}')

                    # Flexible integration for any Python script
                    if wandb_log:
                        # 1. Start a W&B run
                        wandb.init(project=cst.wandb_project, entity=cst.wandb_entity, reinit=True,
                                   group=f'{target_var} - {crop_n} - {month}', config=trial.params,
                                   name=f'{target_var}-{model_type}-{crop_n}-{month}-{hist_norm}',
                                   notes=f'Performance of a 2D CNN model for {target_var} forecasting in Algeria for'
                                         f'crop ID {crop_n}.')

                        # 2. Save model inputs and hyperparameters
                        wandb.config.update({'model_type': model_type,
                                             'crop_n': crop_n,
                                             'month': month,
                                             'norm': hist_norm,
                                             'target': target_var,
                                             'n_epochs': n_epochs,
                                             'batch_size': batch_size,
                                             'n_trials': n_trials
                                             })

                        # Evaluate best model on test set
                        fn_csv_best = [x for x in (dir_tgt / 'best_model').glob('*.csv')][0]
                        res_i = model_evaluation(fn_csv_best, crop_n, month, model_type, fn_asapID2AU, fn_stats90)
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

# EOF
