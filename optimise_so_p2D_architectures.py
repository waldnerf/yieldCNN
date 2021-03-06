#!/usr/bin/python

import os, sys
import argparse
import random
import shutil
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import optuna
import joblib
import random

random.seed(4)

from deeplearning.architecture_complexity_p2D import *
from outputfiles.plot import *
from outputfiles.save import *
from outputfiles.evaluation import *
from sits.readingsits2D import *
import mysrc.constants as cst

def objective_p2DCNN_MISO(trial):
    # 2. Suggest values of the hyperparameters using a trial object.
    nbunits_conv_ = trial.suggest_int('nbunits_conv', 10, 45, step=5)
    kernel_size_ = trial.suggest_int('kernel_size', 2, 5)
    strides_ = trial.suggest_int('strides', 2, 5)
    pool_size_ = trial.suggest_int('pool_size', 1, 5)
    v_fc_ = trial.suggest_categorical('v_fc', [0, 1])
    nbunits_v_ = trial.suggest_int('nbunits_v', 10, 25, step=5)
    nb_fc_ = trial.suggest_categorical('nb_fc', [1, 2])
    funits_fc_ = trial.suggest_categorical('funits_fc', [1, 2, 3])
    activation_ = trial.suggest_categorical('activation', ['relu', 'sigmoid'])

    # Define output filenames
    fn_fig_val = dir_tgt / f'{(out_model).split(".h5")[0]}' \
                           f'_res_{trial.number}_val_{nbunits_conv_}_{kernel_size_}_{strides_}_{pool_size_}_' \
                           f'{v_fc_}_{nbunits_v_}_{nb_fc_}_{funits_fc_}_{activation_}.png'
    fn_fig_test = dir_tgt / f"{(out_model).split('.h5')[0]}" \
                            f'_res_{trial.number}_test_{nbunits_conv_}_{kernel_size_}_{strides_}_{pool_size_}_' \
                            f'{v_fc_}_{nbunits_v_}_{nb_fc_}_{funits_fc_}_{activation_}.png'
    fn_cv_test = dir_tgt / f'{(out_model).split(".h5")[0]}' \
                           f'_res_{trial.number}_test_{nbunits_conv_}_{kernel_size_}_{strides_}_{pool_size_}_' \
                           f'{v_fc_}_{nbunits_v_}_{nb_fc_}_{funits_fc_}_{activation_}.csv'
    out_model_file = dir_tgt / f'{out_model.split(".h5")[0]}_{crop_n}.h5'

    mses_val, r2s_val, mses_test, r2s_test = [], [], [], []
    df_val, df_test_means, df_test_stds, df_details = None, None, None, None
    cv_i = 0
    for test_i in np.unique(groups):
        val_i = random.choice([x for x in np.unique(groups) if x != test_i])
        train_i = [x for x in np.unique(groups) if x != val_i and x != test_i]

        Xt_train, Xv_train, ohe_train, y_train = subset_data(Xt, Xv, region_ohe, y,
                                                             [x in train_i for x in groups])
        Xt_val, Xv_val, ohe_val, y_val = subset_data(Xt, Xv, region_ohe, y, groups == val_i)
        Xt_test, Xv_test, ohe_test, y_test = subset_data(Xt, Xv, region_ohe, y, groups == test_i)

        # ---- Normalizing the data per band
        min_per_t, max_per_t, min_per_v, max_per_v, min_per_y, max_per_y = computingMinMax(Xt_train,
                                                                                           Xv_train,
                                                                                           train_i)
        # Normalise training set
        Xt_train = normalizingData(Xt_train, min_per_t, max_per_t)
        Xv_train = normalizingData(Xv_train, min_per_v, max_per_v)
        # Normalise validation set
        Xt_val = normalizingData(Xt_val, min_per_t, max_per_t)
        Xv_val = normalizingData(Xv_val, min_per_v, max_per_v)
        # Normalise test set
        Xt_test = normalizingData(Xt_test, min_per_t, max_per_t)
        Xv_test = normalizingData(Xv_test, min_per_v, max_per_v)

        # Normalise ys
        transformer_y = MinMaxScaler().fit(y_train[:, [crop_n]])
        ys_train = transformer_y.transform(y_train[:, [crop_n]])
        ys_val = transformer_y.transform(y_val[:, [crop_n]])
        ys_test = transformer_y.transform(y_test[:, [crop_n]])

        # ---- concatenate OHE and Xv
        Xv_train = ohe_train  # np.concatenate([Xv_train[:, [crop_n]], ohe_train], axis=1)
        Xv_val = ohe_val  # np.concatenate([Xv_val[:, [crop_n]], ohe_val], axis=1)
        Xv_test = ohe_test  #np.concatenate([Xv_test[:, [crop_n]], ohe_test], axis=1)

        # We compile our model with a sampled learning rate.
        model = Archi_prob3_2DCNNw_MISO(Xt,
                                       region_ohe,
                                       nbunits_conv=nbunits_conv_,
                                       kernel_size=kernel_size_,
                                       strides=strides_,
                                       pool_size=pool_size_,
                                       v_fc=v_fc_,
                                       nbunits_v=nbunits_v_,
                                       nb_fc=nb_fc_,
                                       funits_fc=funits_fc_,
                                       activation=activation_,
                                       verbose=False)

        model, y_val_means, _ = cv_Model_MISO(model, Xt_train, Xv_train, ys_train, Xt_val, Xv_val, ys_val,
                                           out_model_file, n_epochs=n_epochs, batch_size=batch_size)
        y_val_means = transformer_y.inverse_transform(y_val_means)
        out_val = np.concatenate([y_val[:, [crop_n]], y_val_means], axis=1)

        X_test_inf = {'ts_input': Xt_test, 'v_input': Xv_test}
        y_test_means, y_test_stds = inference_total_uncertainty(model, X_test_inf, scaler_=transformer_y)
        out_test_means = np.concatenate([y_test[:, [crop_n]], y_test_means], axis=1)
        out_test_stds = y_test_stds
        out_details = np.expand_dims(region_id[groups == test_i].T, axis=1)
        if df_val is None:
            df_val = out_val
            df_test_means = out_test_means
            df_test_stds = out_test_stds

            df_details = np.concatenate([out_details, (np.ones_like(out_details) * test_i)], axis=1)
        else:
            df_val = np.concatenate([df_val, out_val], axis=0)
            df_test_means = np.concatenate([df_test_means, out_test_means], axis=0)
            df_test_stds = np.concatenate([df_test_stds, out_test_stds], axis=0)
            df_details = np.concatenate([df_details,
                                         np.concatenate([out_details, (np.ones_like(out_details) * test_i)], axis=1)],
                                        axis=0)

        mse_val = mean_squared_error(y_val[:, [crop_n]], y_val_means, squared=False, multioutput='raw_values')
        r2_val = r2_score(y_val[:, [crop_n]], y_val_means)
        mses_val.append(mse_val)
        r2s_val.append(r2_val)
        mse_test = mean_squared_error(y_test[:, [crop_n]], y_test_means, squared=False, multioutput='raw_values')
        r2_test = r2_score(y_test[:, [crop_n]], y_test_means)
        mses_test.append(mse_test)
        r2s_test.append(r2_test)

        trial.report(np.mean(r2s_val), cv_i)  # report mse
        if trial.should_prune():  # let optuna decide whether to prune
            raise optuna.exceptions.TrialPruned()
        cv_i += 1

    av_rmse_val = np.mean(mses_val)
    av_r2_val = np.mean(r2s_val)
    av_rmse_test = np.mean(mses_test)

    plt.plot([0, 5], [0, 5], '-', color='black')
    plt.plot(df_val[:, 1], df_val[:, 0], '.')
    plt.title(f'RMSE: {np.round(av_rmse_val, 4)} - R^2 = {np.round(np.mean(r2s_val), 4)}')

    plt.xlabel('Predictions (t/ha)')
    plt.ylabel('Observations (t/ha)')
    plt.xlim(0.0, 5.0)
    plt.ylim(0.0, 5.0)

    plt.savefig(fn_fig_val)
    plt.close()

    plt.plot([0, 5], [0, 5], '--', color='black')
    #plt.errorbar(df_test_means[:, 0], df_test_means[:, 1], yerr=df_test_stds[:, 0], fmt='.', color='orange')
    plt.plot(df_test_means[:, 0], df_test_means[:, 1], '.', color='orange')
    plt.title(f'RMSE: {np.round(av_rmse_test, 4)} - R^2 = {np.round(np.mean(r2s_test), 4)}')

    plt.xlabel('Predictions (t/ha)')
    plt.ylabel('Observations (t/ha)')
    plt.xlim(0.0, 5.0)
    plt.ylim(0.0, 5.0)

    plt.savefig(fn_fig_test)
    plt.close()
    # Save CV results
    df_out = np.concatenate([df_details, df_test_means, df_test_stds], axis=1)
    pd.DataFrame(df_out, columns=['ASAP1_ID', 'Year', 'Observed', 'Predicted', 'Stds']).to_csv(fn_cv_test, index=False)

    return av_r2_val


# -----------------------------------------------------------------------
def main(fn_indata, dir_out, model_type='p2DCNN_MISO', overwrite=False):
    # -- Define global variables
    global out_model
    global crop_n
    global Xt
    global n_channels
    global groups
    global Xv
    global region_ohe
    global y
    global n_epochs
    global batch_size
    global region_id
    global dir_tgt

    # ---- Parameters to set
    n_channels = 4  # -- NDVI, Rad, Rain, Temp

    # ---- Get filenames
    print("Input file: ", os.path.basename(str(fn_indata)))

    # ---- output files
    dir_out.mkdir(parents=True, exist_ok=True)
    dir_res = dir_out / f'Archi+{str(model_type)}'
    dir_res.mkdir(parents=True, exist_ok=True)
    print("noarchi: ", model_type)
    out_model = f'archi-{model_type}.h5'

    # ---- Downloading
    Xt_full, Xv, region_id, groups, y = data_reader(fn_indata)

    # ---- Convert region to one hot
    region_ohe = add_one_hot(region_id)

    # ---- Getting train/val/test data

    # ---- variables
    n_epochs = 25
    batch_size = 800
    n_trials = 100

    # loop through all crops
    for crop_n in range(y.shape[1]):
        dir_crop = dir_res / f'crop_{crop_n}'
        dir_crop.mkdir(parents=True, exist_ok=True)
        # loop by month
        for month in range(2, 9):
            dir_tgt = dir_crop / f'month_{month}'
            dir_tgt.mkdir(parents=True, exist_ok=True)

            if (len([x for x in dir_tgt.glob('best_model')]) != 0) & (overwrite is False):
                pass
            else:
                rm_tree(dir_tgt)
                idx = (month + 1) * 3
                Xt = Xt_full[:, :, 0:idx, :]

                study = optuna.create_study(direction='maximize',
                                            pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=5)
                                            )
                if model_type == 'p2DCNN_MISO':
                    study.optimize(objective_p2DCNN_MISO, n_trials=n_trials)
                else:
                    NotImplementedError

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

                joblib.dump(study, os.path.join(dir_tgt, f'study_{crop_n}_{model_type}.dump'))
                # dumped_study = joblib.load(os.path.join(cst.my_project.meta_dir, 'study_in_memory_storage.dump'))
                # dumped_study.trials_dataframe()
                df = study.trials_dataframe().to_csv(os.path.join(dir_tgt, f'study_{crop_n}_{model_type}.csv'))
                # fig = optuna.visualization.plot_slice(study)
                print('------------------------------------------------')

                save_best_model(dir_tgt, f'res_{trial.number}')


# -----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        fn_indata = cst.my_project.data_dir / f'{cst.target}_full_2d_dataset.pickle'
        dir_out = cst.my_project.params_dir
        main(fn_indata, dir_out, overwrite=False)
        print("0")
    except RuntimeError:
        print >> sys.stderr
        sys.exit(1)

# EOF
