#!/usr/bin/python

import os, sys
import argparse
import random
from pycm import *
import shutil
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from deeplearning.architecture_complexity_1d import *
from outputfiles.plot import *
from outputfiles.save import *
from outputfiles.evaluation import *
from sits.readingsits1D import *
import mysrc.constants as cst


# -----------------------------------------------------------------------
def main(fn_indata, sits_path_val, sits_path_test, dir_out, feature, noarchi, norun):
    fn_indata = str(cst.my_project.data_dir / f'{cst.target}_full_dataset.csv')
    dir_out = cst.my_project.params_dir
    feature = 'SB'
    noarchi = 2
    norun = 0

    # ---- Parameters to set
    n_channels = 4  # -- NDVI, Rad, Rain, Temp

    # ---- Evaluated metrics
    eval_label = ['MSE', 'train_loss', 'train_time', 'test_time']

    # ---- Get filenames
    print("Input file: ", os.path.basename(str(fn_indata)))

    # ---- output files
    dir_res = dir_out / f'Archi+{str(noarchi)}'
    dir_res.mkdir(parents=True, exist_ok=True)
    print("noarchi: ", noarchi)
    str_result = f'{feature}-noarchi{noarchi}-norun{norun}'
    res_file = dir_out / f'resultOA-{str_result}.csv'
    res_mat = np.zeros((len(eval_label), 1))
    traintest_loss_file = dir_out / f'trainingHistory-{str_result}.csv'
    out_model_file = dir_out / f'bestmodel-{str_result}.h5'

    # ---- Downloading
    Xt, Xv, region_id, groups, y = data_reader(fn_indata)

    # ---- Convert region to one hot
    region_ohe = add_one_hot(region_id)

    # ---- Get model
    model_type = 'CNNw_SISO'

    switcher = {
        'CNNw_SISO': Archi_CNNw_SISO,
        'CNNw_SIMO': Archi_CNNw_SIMO,
        'CNNw_MIMO': Archi_CNNw_MIMO,
        'CNN_SIMO': Archi_CONV_SIMO,
        'CNN_MIMO': Archi_CONV_MIMO,
        'RNN_SIMO': Archi_RNN_FC_SIMO,
        'RNN_MIMO': Archi_RNN_FC_MIMO
    }

    func = switcher.get(model_type, lambda: 0)

    # ---- Getting train/val/test data
    import random
    random.seed(4)

    df_out = None
    for val_i in np.unique(groups):
        test_i = random.choice([x for x in np.unique(groups) if x != val_i])
        train_i = [x for x in np.unique(groups) if x != val_i and x != test_i]

        Xt_train, Xv_train, ohe_train, y_train = subset_data(Xt, Xv, region_ohe, y, [x in train_i for x in groups])
        Xt_val, Xv_val, ohe_val, y_val = subset_data(Xt, Xv, region_ohe, y, groups == val_i)
        Xt_test, Xv_test, ohe_test, y_test = subset_data(Xt, Xv, region_ohe, y, groups == test_i)

        # ---- Reshaping data necessary
        Xt_train = reshape_data(Xt_train, n_channels)
        Xt_val = reshape_data(Xt_val, n_channels)
        Xt_test = reshape_data(Xt_test, n_channels)

        # ---- Normalizing the data per band
        #minMaxVal_file = '.'.join(str(out_model_file).split('.')[0:-1])
        #minMaxVal_file = minMaxVal_file + '_minMax.txt'
        #if not os.path.exists(minMaxVal_file):
        min_per_t, max_per_t, min_per_v, max_per_v, min_per_y, max_per_y = computingMinMax(Xt_train, Xv_train, train_i)
        #else:
        #    min_per_t, max_per_t, min_per_v, max_per_v = read_minMaxVal(minMaxVal_file)  # TODO!
        #transformer_Xt = RobustScaler(quantile_range=(0.02, 0.98)).fit(Xt_train)

        Xt_train = normalizingData(Xt_train, min_per_t, max_per_t)
        Xv_train = normalizingData(Xv_train, min_per_v, max_per_v)

        Xt_val = normalizingData(Xt_val, min_per_t, max_per_t)
        Xv_val = normalizingData(Xv_val, min_per_v, max_per_v)

        Xt_test = normalizingData(Xt_test, min_per_t, max_per_t)
        Xv_test = normalizingData(Xv_test, min_per_v, max_per_v)

        transformer_y = MinMaxScaler().fit(y_train)
        ys_train = transformer_y.transform(y_train)
        ys_val = transformer_y.transform(y_val)
        ys_test = transformer_y.transform(y_test)

        # ---- concatenate OHE and Xv
        Xv_train = np.concatenate([Xv_train, ohe_train], axis=1)
        Xv_val = np.concatenate([Xv_val, ohe_val], axis=1)
        Xv_test = np.concatenate([Xv_test, ohe_test], axis=1)

        # ---- variables
        n_epochs = 100
        batch_size = 40

        if model_type == 'RNN_MIMO':
            model = func(Xt_train, Xv_train, nb_rnn=1, nbunits_rnn=32, nbunits_fc=32, verbose=False)
            tmodel, mse1, mse2, mse3, history, train_time, test_time = \
                trainValTestModel_MIMO(model, Xt_train, Xv_train, ys_train, Xt_val, Xv_val, ys_val, Xt_test, Xv_test,
                                       ys_test, out_model_file, n_epochs=n_epochs, batch_size=batch_size)
            pred1, pred2, pred3 = tmodel.predict(x={'ts_input': Xt_test, 'v_input': Xv_test})

            pred1, pred2, pred3 = tmodel.predict(x={'ts_input': Xt_test, 'v_input': Xv_test})
            plot_predictions(ys_test,  np.concatenate([pred1, pred2, pred3], axis=1), title='Test')

            pred1, pred2, pred3 = tmodel.predict(x={'ts_input': Xt_train, 'v_input': Xv_train})
            plot_predictions(ys_train,  np.concatenate([pred1, pred2, pred3], axis=1), title='Train')

        elif model_type == 'RNN_SIMO':
            model = func(Xt_train, nb_rnn=1, nbunits_rnn=18, nbunits_fc=32, verbose=False)
            tmodel, mse1, mse2, mse3, history, train_time, test_time = \
                trainValTestModel_SIMO(model, Xt_train, ys_train, Xt_val, ys_val, Xt_test,
                                       ys_test, out_model_file, n_epochs=n_epochs, batch_size=batch_size)
            pred1, pred2, pred3 = tmodel.predict(x={'ts_input': Xt_test})

        elif model_type == 'CNN_MIMO':
            model = func(Xt_train, Xv_train, nb_conv=1, nbunits_conv=16, nbunits_fc=32, verbose=False)
            tmodel, mse1, mse2, mse3, history, train_time, test_time = \
                trainValTestModel_MIMO(model, Xt_train, Xv_train, ys_train, Xt_val, Xv_val, ys_val, Xt_test, Xv_test,
                                       ys_test, out_model_file, n_epochs=n_epochs, batch_size=batch_size)
            pred1, pred2, pred3 = tmodel.predict(x={'ts_input': Xt_test, 'v_input': Xv_test})


        elif  model_type == 'CNN_SIMO':
            model = func(Xt_train, nb_conv=4, nbunits_conv=5, nbunits_fc=32, verbose=False)
            tmodel, mse1, mse2, mse3, history, train_time, test_time = \
                trainValTestModel_SIMO(model, Xt_train, ys_train, Xt_val, ys_val, Xt_test,
                                       ys_test, out_model_file, n_epochs=n_epochs, batch_size=batch_size)

            pred1, pred2, pred3 = tmodel.predict(x={'ts_input': Xt_test})
            plot_predictions(ys_test,  np.concatenate([pred1, pred2, pred3], axis=1), title='Test')

        elif  model_type == 'CNNw_SIMO':
            model = func(Xt_train, nbunits_conv=15, verbose=False)
            tmodel, mse1, mse2, mse3, history, train_time, test_time = \
                trainValTestModel_SIMO(model, Xt_train, ys_train, Xt_val, ys_val, Xt_test,
                                       ys_test, out_model_file, n_epochs=n_epochs, batch_size=batch_size)

            pred1, pred2, pred3 = tmodel.predict(x={'ts_input': Xt_test})
            #plot_predictions(ys_test, np.concatenate([pred1, pred2, pred3], axis=1), title='Test')

        elif model_type == 'CNNw_MIMO':
            model = func(Xt_train,  Xv_train, nbunits_conv=10, verbose=False)
            tmodel, mse1, mse2, mse3, history, train_time, test_time = \
                trainValTestModel_MIMO(model, Xt_train, Xv_train, ys_train, Xt_val, Xv_val, ys_val, Xt_test, Xv_test,
                                       ys_test, out_model_file, n_epochs=n_epochs, batch_size=batch_size)

            pred1, pred2, pred3 = tmodel.predict(x={'ts_input': Xt_test, 'v_input': Xv_test})

        elif model_type == 'CNNw_SISO':
            model = func(Xt_train,  nbunits_conv=10, verbose=False)
            tmodel, mse1, history, train_time, test_time = \
                trainValTestModel_SISO(model, Xt_train, ys_train[:, [0]], Xt_val, ys_val[:, [0]], Xt_test,
                                       ys_test[:, [0]], out_model_file, n_epochs=n_epochs, batch_size=batch_size)
            pred1 = tmodel.predict(x={'ts_input': Xt_test})

        preds = transformer_y.inverse_transform(np.concatenate([pred1, pred2, pred3], axis=1))

        out_i = np.concatenate([y_test, preds], axis=1)
        if df_out is None:
            df_out = out_i
        else:
            df_out = np.concatenate([df_out, out_i], axis=0)

    plt.plot(df_out[:,0], df_out[:,3], '.')

    preds = np.concatenate([pred1, pred2, pred3], axis=1)
    out_i = np.concatenate([ys_test, preds], axis=1)
    #df_out = out_i
    plot_predictions(df_out[:,0:3], df_out[:,3:6], title='Test')
    mean_squared_error(df_out[:, 0:3],  df_out[:, 3:6], squared=False, multioutput='raw_values')

    # saveLossAcc(model_hist, traintest_loss_file)

    # print('Overall accuracy (OA): ', res_mat[0, norun])
    # print('Train loss: ', res_mat[1, norun])
    print('Training time (s): ', res_mat[2, norun])
    print('Test time (s): ', res_mat[3, norun])

    # ---- saving res_file
    # saveMatrix(np.transpose(res_mat), res_file, eval_label)


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
