#!/usr/bin/python

import os, sys
import argparse
import random
import shutil
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import optuna
import joblib

from deeplearning.architecture_complexity_1D import *
from outputfiles.plot import *
from outputfiles.save import *
from outputfiles.evaluation import *
from sits.readingsits1D import *
import mysrc.constants as cst


def objective_CNNw_SIMO(trial):
    # 2. Suggest values of the hyperparameters using a trial object.
    nbunits_conv_ = trial.suggest_int('nbunits_conv', 10, 50)
    kernel_size_ = trial.suggest_int('kernel_size', 2, 5)
    strides_ = trial.suggest_int('strides', 2, 5)
    pool_size_ = trial.suggest_int('pool_size', 2, 5)
    dropout_rate_ = trial.suggest_float('dropout_rate', 0, 0.2, step=0.05)
    nb_fc_ =  trial.suggest_categorical('nb_fc_', [1, 2, 3])
    funits_fc_ = trial.suggest_categorical('funits_fc', [2, 3, 4])
    activation_ = trial.suggest_categorical('activation', ['relu', 'sigmoid'])

    # Define output filenames

    fn_fig_val = dir_res / f'{(out_model).split(".h5")[0]}' \
                           f'_res_{trial.number}_val_{nbunits_conv_}_{kernel_size_}_{strides_}_{pool_size_}_' \
                           f'{round(dropout_rate_ * 100)}_{funits_fc_}_{funits_fc_}_{activation_}.png'
    fn_fig_test = dir_res / f"{(out_model).split('.h5')[0]}" \
                            f'_res_{trial.number}_test_{nbunits_conv_}_{kernel_size_}_{strides_}_{pool_size_}_' \
                            f'{round(dropout_rate_ * 100)}_{funits_fc_}_{funits_fc_}_{activation_}.png'
    fn_cv_test = dir_res / f'{(out_model).split(".h5")[0]}' \
                           f'_res_{trial.number}_test_{nbunits_conv_}_{kernel_size_}_{strides_}_{pool_size_}_' \
                           f'{round(dropout_rate_ * 100)}_{funits_fc_}_{funits_fc_}_{activation_}.csv'
    out_model_file = dir_res / f'{out_model.split(".h5")[0]}.h5'

    # Generate Model
    model = Archi_CNNw_SIMO_mt([Xt.shape[1] // n_channels, n_channels],
                               nbunits_conv=nbunits_conv_,
                               kernel_size=kernel_size_,
                               strides=strides_,
                               pool_size=pool_size_,
                               dropout_rate=dropout_rate_,
                               funits_fc=funits_fc_,
                               nb_fc=nb_fc_,
                               activation=activation_,
                               verbose=False)
    mses_val, r2s_val, mses_test, r2s_test = [], [], [], []
    df_val, df_test, df_details = None, None, None
    for test_i in np.unique(groups):
        val_i = random.choice([x for x in np.unique(groups) if x != test_i])
        train_i = [x for x in np.unique(groups) if x != val_i and x != test_i]

        Xt_train, Xv_train, ohe_train, y_train = subset_data(Xt, Xv, region_ohe, y,
                                                             [x in train_i for x in groups])
        Xt_val, Xv_val, ohe_val, y_val = subset_data(Xt, Xv, region_ohe, y, groups == val_i)
        Xt_test, Xv_test, ohe_test, y_test = subset_data(Xt, Xv, region_ohe, y, groups == test_i)

        # ---- Reshaping data necessary
        Xt_train = reshape_data(Xt_train, n_channels)
        Xt_val = reshape_data(Xt_val, n_channels)
        Xt_test = reshape_data(Xt_test, n_channels)

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
        transformer_y = MinMaxScaler().fit(y_train)
        ys_train = transformer_y.transform(y_train)
        ys_val = transformer_y.transform(y_val)
        ys_test = transformer_y.transform(y_test)

        # We compile our model with a sampled learning rate.
        model, y_val_preds = cv_Model_SIMO_mt(model, Xt_train, ys_train, Xt_val, ys_val,
                                           out_model_file, n_epochs=n_epochs, batch_size=batch_size)
        y_val_preds = np.squeeze(np.stack(y_val_preds)).T
        y_val_preds = transformer_y.inverse_transform(y_val_preds)
        out_val = np.concatenate([y_val, y_val_preds], axis=1)

        y_test_preds = model.predict(x={'ts_input': Xt_test})
        y_test_preds = np.squeeze(np.stack(y_test_preds)).T
        y_test_preds = transformer_y.inverse_transform(y_test_preds)
        out_test = np.concatenate([y_test, y_test_preds], axis=1)
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

        mse_val = mean_squared_error(y_val, y_val_preds, squared=False, multioutput='raw_values')
        r2_val = r2_score(y_val, y_val_preds, multioutput='raw_values')
        mses_val.append(mse_val)
        r2s_val.append(r2_val)
        mse_test = mean_squared_error(y_test, y_test_preds, squared=False, multioutput='raw_values')
        r2_test = r2_score(y_test, y_test_preds)
        mses_test.append(mse_test)
        r2s_test.append(r2_test)

    av_rmse_val = np.round(np.mean(np.stack(mses_val, axis=0), axis=0), 4)
    av_rmse_test = np.round(np.mean(np.stack(mses_test, axis=0), axis=0), 4)
    av_r2_val = np.round(np.mean(np.stack(r2s_val, axis=0), axis=0), 4)
    av_r2_test = np.round(np.mean(np.stack(r2s_test, axis=0), axis=0), 4)

    plot_predictions_mo(df_val[:, 0:3], df_val[:, 3:6], fn=fn_fig_val)
    plot_predictions_mo(df_test[:, 0:3], df_test[:, 3:6], fn=fn_fig_test)

    # Save CV results
    df_out = np.concatenate([df_details, df_test], axis=1)
    pd.DataFrame(df_out, columns=['ASAP1_ID', 'Year', 'Crop0_Observed', 'Crop0_Predicted',
                                  'Crop1_Observed', 'Crop1_Predicted', 'Crop2_Observed', 'Crop2_Predicted']).\
        to_csv(fn_cv_test, index=False)

    return np.mean(av_rmse_val)



def objective_CNNw_MIMO(trial):
    # 2. Suggest values of the hyperparameters using a trial object.
    nbunits_conv_ = trial.suggest_int('nbunits_conv', 10, 50)
    kernel_size_ = trial.suggest_int('kernel_size', 2, 5)
    strides_ = trial.suggest_int('strides', 2, 5)
    pool_size_ = trial.suggest_int('pool_size', 2, 5)
    dropout_rate_ = trial.suggest_float('dropout_rate', 0, 0.2, step=0.05)
    nb_fc_ = trial.suggest_categorical('nb_fc_', [1, 2, 3])
    funits_fc_ = trial.suggest_categorical('funits_fc', [2, 3, 4, 5])
    activation_ = trial.suggest_categorical('activation', ['relu', 'sigmoid'])

    # Define output filenames

    fn_fig_val = dir_res / f'{(out_model).split(".h5")[0]}' \
                           f'_res_{trial.number}_val_{nbunits_conv_}_{kernel_size_}_{strides_}_{pool_size_}_' \
                           f'{round(dropout_rate_ * 100)}_{funits_fc_}_{funits_fc_}_{activation_}.png'
    fn_fig_test = dir_res / f"{(out_model).split('.h5')[0]}" \
                            f'_res_{trial.number}_test_{nbunits_conv_}_{kernel_size_}_{strides_}_{pool_size_}_' \
                            f'{round(dropout_rate_ * 100)}_{funits_fc_}_{funits_fc_}_{activation_}.png'
    fn_cv_test = dir_res / f'{(out_model).split(".h5")[0]}' \
                           f'_res_{trial.number}_test_{nbunits_conv_}_{kernel_size_}_{strides_}_{pool_size_}_' \
                           f'{round(dropout_rate_ * 100)}_{funits_fc_}_{funits_fc_}_{activation_}.csv'
    out_model_file = dir_res / f'{out_model.split(".h5")[0]}.h5'

    # Generate Model
    model = Archi_CNNw_SIMO_mt([Xt.shape[1] // n_channels, n_channels],
                               nbunits_conv=nbunits_conv_,
                               kernel_size=kernel_size_,
                               strides=strides_,
                               pool_size=pool_size_,
                               dropout_rate=dropout_rate_,
                               funits_fc=funits_fc_,
                               nb_fc=nb_fc_,
                               activation=activation_,
                               verbose=False)
    mses_val, r2s_val, mses_test, r2s_test = [], [], [], []
    df_val, df_test, df_details = None, None, None
    for test_i in np.unique(groups):
        val_i = random.choice([x for x in np.unique(groups) if x != test_i])
        train_i = [x for x in np.unique(groups) if x != val_i and x != test_i]

        Xt_train, Xv_train, ohe_train, y_train = subset_data(Xt, Xv, region_ohe, y,
                                                             [x in train_i for x in groups])
        Xt_val, Xv_val, ohe_val, y_val = subset_data(Xt, Xv, region_ohe, y, groups == val_i)
        Xt_test, Xv_test, ohe_test, y_test = subset_data(Xt, Xv, region_ohe, y, groups == test_i)

        # ---- Reshaping data necessary
        Xt_train = reshape_data(Xt_train, n_channels)
        Xt_val = reshape_data(Xt_val, n_channels)
        Xt_test = reshape_data(Xt_test, n_channels)

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
        transformer_y = MinMaxScaler().fit(y_train)
        ys_train = transformer_y.transform(y_train)
        ys_val = transformer_y.transform(y_val)
        ys_test = transformer_y.transform(y_test)

        # We compile our model with a sampled learning rate.
        model, y_val_preds = cv_Model_MIMO(model, Xt_train, Xv_train, ys_train, Xt_val, Xv_val, ys_val,
                                           out_model_file, n_epochs=n_epochs, batch_size=batch_size)
        y_val_preds = np.squeeze(np.stack(y_val_preds)).T
        y_val_preds = transformer_y.inverse_transform(y_val_preds)
        out_val = np.concatenate([y_val, y_val_preds], axis=1)

        y_test_preds = model.predict(x={'ts_input': Xt_test, 'v_input': Xv_test})
        y_test_preds = np.squeeze(np.stack(y_test_preds)).T
        y_test_preds = transformer_y.inverse_transform(y_test_preds)
        out_test = np.concatenate([y_test, y_test_preds], axis=1)
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

        mse_val = mean_squared_error(y_val, y_val_preds, squared=False, multioutput='raw_values')
        r2_val = r2_score(y_val, y_val_preds, multioutput='raw_values')
        mses_val.append(mse_val)
        r2s_val.append(r2_val)
        mse_test = mean_squared_error(y_test, y_test_preds, squared=False, multioutput='raw_values')
        r2_test = r2_score(y_test, y_test_preds)
        mses_test.append(mse_test)
        r2s_test.append(r2_test)

    av_rmse_val = np.round(np.mean(np.stack(mses_val, axis=0), axis=0), 4)
    av_rmse_test = np.round(np.mean(np.stack(mses_test, axis=0), axis=0), 4)
    av_r2_val = np.round(np.mean(np.stack(r2s_val, axis=0), axis=0), 4)
    av_r2_test = np.round(np.mean(np.stack(r2s_test, axis=0), axis=0), 4)

    plot_predictions_mo(df_val[:, 0:3], df_val[:, 3:6], fn=fn_fig_val)
    plot_predictions_mo(df_test[:, 0:3], df_test[:, 3:6], fn=fn_fig_test)

    # Save CV results
    df_out = np.concatenate([df_details, df_test], axis=1)
    pd.DataFrame(df_out, columns=['ASAP1_ID', 'Year', 'Crop0_Observed', 'Crop0_Predicted',
                                  'Crop1_Observed', 'Crop1_Predicted', 'Crop2_Observed', 'Crop2_Predicted']).\
        to_csv(fn_cv_test, index=False)

    return np.mean(av_rmse_val)

# -----------------------------------------------------------------------
def main(fn_indata, dir_out, model_type='CNNw_SIMO'):
    fn_indata = str(cst.my_project.data_dir / f'{cst.target}_full_dataset.csv')
    dir_out = cst.my_project.params_dir
    dir_out.mkdir(parents=True, exist_ok=True)

    # ---- Parameters to set
    n_channels = 4  # -- NDVI, Rad, Rain, Temp

    # ---- Get filenames
    print("Input file: ", os.path.basename(str(fn_indata)))

    # ---- output files
    dir_res = dir_out / f'Archi+{str(model_type)}'
    dir_res.mkdir(parents=True, exist_ok=True)
    rm_tree(dir_res)
    print("noarchi: ", model_type)
    out_model = f'archi-{model_type}.h5'

    # ---- Downloading
    Xt, Xv, region_id, groups, y = data_reader(fn_indata)

    # ---- Convert region to one hot
    region_ohe = add_one_hot(region_id)

    # ---- Getting train/val/test data
    import random
    random.seed(4)

    # ---- variables
    n_epochs = 80
    batch_size = 800
    n_trials = 40

    #
    study = optuna.create_study(direction='minimize')
    if model_type == 'CNNw_SIMO':
        study.optimize(objective_CNNw_SIMO, n_trials=n_trials)
    elif model_type == 'CNNw_MIMO':
        study.optimize(objective_CNNw_SIMO, n_trials=n_trials)

    trial = study.best_trial
    print('------------------------------------------------')
    print('--------------- Optimisation results -----------')
    print('------------------------------------------------')
    print("Number of finished trials: ", len(study.trials))
    print(f"\n           Best trial ({trial.number})        \n")
    print("MSE: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print("{}: {}".format(key, value))

    joblib.dump(study, os.path.join(cst.my_project.params_dir, f'study_allcrops_{model_type}.dump'))
    df = study.trials_dataframe().to_csv(
        os.path.join(cst.my_project.params_dir, f'study_allcrops_{model_type}.csv'))
    # fig = optuna.visualization.plot_slice(study)
    print('------------------------------------------------')

    save_best_model(dir_res, f'res_{trial.number}')


# -----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # m sits_path, res_path, feature, noarchi, norun
        main(str(cst.my_project.train_dir),
             str(cst.my_project.val_dir),
             str(cst.my_project.test_dir),
             str(cst.my_project.params_dir), 'SB', 2, 0)
        print("0")
    except RuntimeError:
        print >> sys.stderr
        sys.exit(1)

# EOF
