#!/usr/bin/python

"""
Defining tensorflow.keras training the models
"""

import os
import numpy as np
import time
import os.path

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, History, ReduceLROnPlateau, TensorBoard, EarlyStopping
from tensorflow.keras.models import load_model
import mysrc.constants as cst
import json
import datetime
import global_variables


def cv_Model(model, X_train, ys_train, X_val, ys_val, out_model_file, **train_params):
    """
    Model fitting
    :param model: keras model
    :param X_train: Traning data. Should be a list containing ts_input (time series inputs)
     and/or v_input (vector inputs)
    :param ys_train: Scaled target data
    :param X_val: Validation data. Should be a list containing ts_input (time series inputs)
     and/o v_input (vector inputs)
    :param ys_val: Scaled target data
    :param out_model_file: model filename
    :param train_params: parameters for training
    :return:
    """
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.01)

    # ---- optimizer
    opt = optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    # ---- monitoring the minimum validation loss
    checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss',
                                 verbose=0, save_best_only=True, mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=5, min_lr=0.00001)

    callback_list = [checkpoint, reduce_lr]
    model.compile(optimizer=opt,
                  loss={'out1': 'mse'},
                  loss_weights={'out1': 1.},
                  metrics=['mse'])
    #model.save_weights(cst.root_dir / 'model.h5')
    if global_variables.init_weights == None:
        global_variables.init_weights = model.get_weights()
    else:
        model.set_weights(global_variables.init_weights)

    model_hist = model.fit(X_train,
                           {'out1': ys_train},
                           epochs=n_epochs,
                           batch_size=batch_size, shuffle=True, #TODO: put back shuffle=True
                           validation_data=(X_val, {'out1': ys_val}),
                           verbose=0, callbacks=callback_list) #evrbose=0, callbacks=callback_list)
    # 2021 09 24 Model fit results sometimes (due to Optuna assignation of hypers with given data)
    # in los always equal to NaN (maybe because of exploding gradients, to be checked).
    # Because of this no model is saved and the program was crashing. Now if it happens we save some info and then we return
    # Nan. This should end up in a trial statistics that is NaN that is ignored by Optuna
    if os.path.exists(out_model_file) == False:
        fn = cst.root_dir / f'model_errors.log'
        with open(fn, 'a') as f:
            f.write('\n' + 'architecture_features.py, no model file will generate an error. Printing info: ' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            print('************************************************')
            f.write('\n' + 'history of val_mse')
            f.write(('\n' + ", ".join(map(str, model_hist.history['val_mse']))))
            # Data check section (nan and min max)
            f.write('\n' + "X_train sum (finite if not nan there), min, max")
            f.write(('\n' + ", ".join(map(str, [X_train['ts_input'].sum(), X_train['ts_input'].min(), X_train['ts_input'].max()]))))
            f.write('\n' + "ys_train sum (finite if not nan there), min, max")
            f.write(('\n' + ", ".join(map(str, [ys_train.sum(), ys_train.min(), ys_train.max()]))))
            f.write('\n'+"X_val sum (finite if not nan there), min, max")
            f.write(('\n' + ", ".join(map(str, [X_val['ts_input'].sum(), X_val['ts_input'].min(), X_val['ts_input'].max()]))))
            f.write('\n'+"ys_val sum (finite if not nan there), min, max")
            f.write(('\n' + ", ".join(map(str, [ys_val.sum(), ys_val.min(), ys_val.max()]))))
            #hyper check
            # for i, l in enumerate(model.layers):
            #     print(i, l)
            #     print(l.get_config())
            f.write('\n'+'Hypers suggested by Optuna:')
            n_dense_before_output = (len(model.layers) - 1 - 14 - 1) / 2
            hp_dic = {'cn_fc4Xv_units': str(model.layers[1].get_config()['filters']),
                      'cn kernel_size': str(model.layers[1].get_config()['kernel_size']),
                      'cn strides (fixed)': str(model.layers[1].get_config()['strides']),
                      'cn drop out rate:': str(model.layers[4].get_config()['rate']),
                      'AveragePooling2D pool_size': str(model.layers[5].get_config()['pool_size']),
                      'AveragePooling2D strides': str(model.layers[5].get_config()['strides']),
                      'Input shape': str(model.layers[0].output_shape),
                      'Output shape before Pyramid': str(model.layers[9].output_shape),
                      'SpatialPyramidPooling2D bins': str(model.layers[10].get_config()['bins']),
                      'n FC layers before output (nb_fc)': str(int(n_dense_before_output))
                      }
            for i in range(int(n_dense_before_output)):
                hp_dic[str(i) + ' ' + 'fc_units'] = str(model.layers[15 + i * 2].get_config()['units'])
                hp_dic[str(i) + ' ' + 'drop out rate'] = str(model.layers[16 + i * 2].get_config()['rate'])
            hp_dic['Fit final mse'] = model_hist.history['val_mse'][-1]
            f.write('\n')
            for key, value in hp_dic.items():
                f.write('%s:%s\n' % (key, value))
           # f.write(json.dumps(hp_dic))
            f.write('************************************************')
            #now check some metrics with tensorboard
        # log_dir = cst.root_dir / 'tensorboard_logs' / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # os.makedirs(log_dir, exist_ok=True)
        # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        # callback_list = [checkpoint, reduce_lr, tensorboard_callback]
        # model.load_weights(cst.root_dir / 'model.h5')
        # model_hist = model.fit(X_train,
        #                        {'out1': ys_train},
        #                        epochs=n_epochs,
        #                        batch_size=batch_size, shuffle=True,
        #                        validation_data=(X_val, {'out1': ys_val}),
        #                        verbose=2, callbacks=callback_list)
        # # then tensorboard --logdir=D:\PY_data\leanyf\tensorboard_logs\20210927-152409 --host localhost --port 8088
        # # http://localhost:8088
        return model, ys_val*np.nan
    else:
        del model
        model = load_model(out_model_file)
        pred = model.predict(x=X_val)
        return model, pred



def trainTestModel(model, Xt_train, y_train, X_test, Y_test_onehot, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.0)

    def negative_loglikelihood(targets, estimated_distribution):
        return -estimated_distribution.log_prob(targets)

    # ---- optimizer
    opt = optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
    model.compile(optimizer=opt,
                  loss={'out1': 'mse'},  # , 'out2': 'mse', 'out3': 'mse'},
                  # loss_weights={'out1': 1., 'out2': 1., 'out3': 1.},
                  metrics=['mse'])

    # ---- monitoring the minimum loss
    checkpoint = ModelCheckpoint(out_model_file, monitor='loss',
                                 verbose=0, save_best_only=True, mode='min')
    callback_list = [checkpoint]

    start_train_time = time.time()
    hist = model.fit({'cnn_input': Xt_train, 'v_input': Xv_train},
                     {'out1': ys_train[:, 0]},  # , 'out2': ys_train[:, 1], 'out3': ys_train[:, 2]},
                     epochs=n_epochs,
                     batch_size=batch_size, shuffle=True,
                     # validation_data=(Xt_val, Y_test_onehot),
                     verbose=1, callbacks=callback_list)
    train_time = round(time.time() - start_train_time, 2)

    # -- download the best model
    del model
    model = load_model(out_model_file)
    start_test_time = time.time()
    mdl_ev = model.evaluate(x={'cnn_input': Xt_val, 'v_input': Xv_val},
                            y={'out1': ys_val[:, 0], 'out2': ys_val[:, 1], 'out3': ys_val[:, 2]},
                            batch_size=128, verbose=0)
    # print(model.metrics_names)
    loss, loss1, loss2, loss3, mse1, mse2, mse3 = mdl_ev

    test_time = round(time.time() - start_test_time, 2)

    return test_acc, np.min(hist.history['loss']), model, hist.history, train_time, test_time


# -----------------------------------------------------------------------
def cv_Model_SISO(model, Xt_train, ys_train, Xt_val, ys_val, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.01)

    # ---- optimizer
    opt = optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    # ---- monitoring the minimum validation loss
    checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss',
                                 verbose=0, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    model.compile(optimizer=opt,
                  loss={'out1': 'mse'},
                  loss_weights={'out1': 1.},
                  metrics=['mse'])

    hist = model.fit(Xt_train,
                     {'out1': ys_train},
                     epochs=n_epochs,
                     batch_size=batch_size, shuffle=True,
                     validation_data=(Xt_val,
                                      {'out1': ys_val}),
                     verbose=0, callbacks=callback_list)

    del model
    model = load_model(out_model_file)
    pred = model.predict(x={'ts_input': Xt_val})
    return model, pred


# -----------------------------------------------------------------------
def cv_Model_MIMO(model, Xt_train, Xv_train, ys_train, Xt_val, Xv_val, ys_val, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.01)

    # ---- optimizer
    opt = optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    # ---- monitoring the minimum validation loss
    checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss',
                                 verbose=0, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    model.compile(optimizer=opt,
                  loss={'out1': 'mse', 'out2': 'mse', 'out3': 'mse'},
                  loss_weights={'out1': 1., 'out2': 1., 'out3': 1.},
                  metrics=['mse'])

    hist = model.fit({'ts_input': Xt_train, 'v_input': Xv_train},
                     {'out1': ys_train[:, [0]], 'out2': ys_train[:, [1]], 'out3': ys_train[:, [2]]},
                     epochs=n_epochs,
                     batch_size=batch_size, shuffle=True,
                     validation_data=(Xt_val,
                                      {'out1': ys_val[:, [0]], 'out2': ys_val[:, [1]], 'out3': ys_val[:, [2]]}),
                     verbose=0, callbacks=callback_list)

    del model
    model = load_model(out_model_file)
    pred = model.predict(x={'ts_input': Xt_val, 'v_input': Xv_val})
    return model, pred


# -----------------------------------------------------------------------
def cv_Model_MISO(model, Xt_train, Xv_train, ys_train, Xt_val, Xv_val, ys_val, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.01)

    # ---- optimizer
    opt = optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    # ---- monitoring the minimum validation loss
    checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss',
                                 verbose=0, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    model.compile(optimizer=opt,
                  loss={'out1': 'mse'},
                  loss_weights={'out1': 1.},
                  metrics=['mse'])

    model_hist = model.fit({'ts_input': Xt_train, 'v_input': Xv_train},
                           {'out1': ys_train},
                           epochs=n_epochs,
                           batch_size=batch_size, shuffle=True,
                           validation_data=({'ts_input': Xt_val, 'v_input': Xv_val}, {'out1': ys_val}),
                           verbose=0, callbacks=callback_list)

    del model
    model = load_model(out_model_file)
    pred = model.predict(x={'ts_input': Xt_val, 'v_input': Xv_val})
    return model, pred


# -----------------------------------------------------------------------
def trainValTestModel_SISO(model, Xt_train, ys_train, Xt_val, ys_val,
                           Xt_test, ys_test, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.01)

    # ---- optimizer
    opt = optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    # ---- monitoring the minimum validation loss
    checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss',
                                 verbose=0, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    model.compile(optimizer=opt,
                  loss={'out1': 'mse'},
                  loss_weights={'out1': 1.},
                  metrics=['mse'])
    start_train_time = time.time()
    hist = model.fit(Xt_train,
                     {'out1': ys_train[:, 0]},
                     epochs=n_epochs,
                     batch_size=batch_size, shuffle=True,
                     validation_data=(Xt_val,
                                      {'out1': ys_val[:, 0]}),
                     verbose=1, callbacks=callback_list)

    train_time = round(time.time() - start_train_time, 2)

    # -- download the best model
    del model
    model = load_model(out_model_file)
    start_test_time = time.time()
    mdl_ev = model.evaluate(x={'ts_input': Xt_test},
                            y={'out1': ys_test[:, 0]},
                            batch_size=128, verbose=0)
    # print(model.metrics_names)
    loss, mse1, = mdl_ev

    test_time = round(time.time() - start_test_time, 2)
    return model, mse1, hist.history, train_time, test_time


# -----------------------------------------------------------------------
def trainValTestModel_SIMO(model, Xt_train, ys_train, Xt_val, ys_val,
                           Xt_test, ys_test, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.0)

    # ---- optimizer
    opt = optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    # ---- monitoring the minimum validation loss
    checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss',
                                 verbose=0, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    model.compile(optimizer=opt,
                  loss={'out1': 'mse', 'out2': 'mse', 'out3': 'mse'},
                  loss_weights={'out1': 1., 'out2': 1., 'out3': 1.},
                  metrics=['mse'])
    start_train_time = time.time()
    hist = model.fit(Xt_train,
                     {'out1': ys_train[:, 0], 'out2': ys_train[:, 1], 'out3': ys_train[:, 2]},
                     epochs=n_epochs,
                     batch_size=batch_size, shuffle=True,
                     validation_data=(Xt_val,
                                      {'out1': ys_val[:, 0], 'out2': ys_val[:, 1], 'out3': ys_val[:, 2]}),
                     verbose=1, callbacks=callback_list)

    train_time = round(time.time() - start_train_time, 2)

    # -- download the best model
    del model
    model = load_model(out_model_file)
    start_test_time = time.time()
    mdl_ev = model.evaluate(x={'ts_input': Xt_test},
                            y={'out1': ys_test[:, 0], 'out2': ys_test[:, 1], 'out3': ys_test[:, 2]},
                            batch_size=128, verbose=0)
    # print(model.metrics_names)
    loss, loss1, loss2, loss3, mse1, mse2, mse3 = mdl_ev

    test_time = round(time.time() - start_test_time, 2)
    return model, mse1, mse2, mse3, hist.history, train_time, test_time


# -----------------------------------------------------------------------
def trainValTestModel_MIMO(model, Xt_train, Xv_train, ys_train, Xt_val, Xv_val, ys_val,
                           Xt_test, Xv_test, ys_test, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.0)

    # ---- optimizer
    opt = optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    # ---- monitoring the minimum validation loss
    checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss',
                                 verbose=0, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    model.compile(optimizer=opt,
                  loss={'out1': 'mse', 'out2': 'mse', 'out3': 'mse'},
                  loss_weights={'out1': 1., 'out2': 1., 'out3': 1.},
                  metrics=['mse'])
    start_train_time = time.time()
    hist = model.fit({'ts_input': Xt_train, 'v_input': Xv_train},
                     {'out1': ys_train[:, 0], 'out2': ys_train[:, 1], 'out3': ys_train[:, 2]},
                     epochs=n_epochs,
                     batch_size=batch_size, shuffle=True,
                     validation_data=({'ts_input': Xt_val, 'v_input': Xv_val},
                                      {'out1': ys_val[:, 0], 'out2': ys_val[:, 1], 'out3': ys_val[:, 2]}),
                     verbose=1, callbacks=callback_list)

    train_time = round(time.time() - start_train_time, 2)

    # -- download the best model
    del model
    model = load_model(out_model_file)
    start_test_time = time.time()
    mdl_ev = model.evaluate(x={'ts_input': Xt_test, 'v_input': Xv_test},
                            y={'out1': ys_test[:, 0], 'out2': ys_test[:, 1], 'out3': ys_test[:, 2]},
                            batch_size=128, verbose=0)
    # print(model.metrics_names)
    loss, loss1, loss2, loss3, mse1, mse2, mse3 = mdl_ev

    test_time = round(time.time() - start_test_time, 2)
    return model, mse1, mse2, mse3, hist.history, train_time, test_time


# -----------------------------------------------------------------------
def trainValTestModel_singletask(model, X_train, Y_train_onehot, X_val, Y_val_onehot, X_test, Y_test_onehot,
                                 out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.0)

    # ---- optimizer
    opt = optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
    model.compile(optimizer=opt,
                  loss={'out1': 'mse', 'out2': 'mse'},
                  loss_weights={'out1': 1., 'out2': 1.})

    # ---- monitoring the minimum validation loss
    checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss',
                                 verbose=0, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    callback_list = [checkpoint, early_stop]

    start_train_time = time.time()
    hist = model.fit({'cnn_input': X_train},
                     {'out1': Y_train_onehot, 'out2': Y_train_onehot},
                     epochs=n_epochs,
                     batch_size=batch_size, shuffle=True,
                     validation_data=(X_val, Y_val_onehot),
                     verbose=1, callbacks=callback_list)
    train_time = round(time.time() - start_train_time, 2)

    # -- download the best model
    del model
    model = load_model(out_model_file)
    start_test_time = time.time()
    test_loss, test_acc = model.evaluate(x=X_test, y=Y_test_onehot,
                                         batch_size=128, verbose=0)
    test_time = round(time.time() - start_test_time, 2)

    return test_acc, np.min(hist.history['val_loss']), model, hist.history, train_time, test_time

# EOF
