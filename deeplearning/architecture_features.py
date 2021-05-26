#!/usr/bin/python

""" 
	Defining keras architecture, and training the models
"""

import sys, os
import numpy as np
import time

import keras
from keras import layers
from keras import optimizers
from keras.regularizers import l2
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Flatten, Lambda, SpatialDropout1D, \
    Concatenate
from keras.layers import Conv1D, Conv2D, AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import Callback, ModelCheckpoint, History, EarlyStopping
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from keras import backend as K


# -----------------------------------------------------------------------
# ---------------------- Modules
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
def conv_bn(X, **conv_params):
    nbunits = conv_params["nbunits"];
    kernel_size = conv_params["kernel_size"];

    strides = conv_params.setdefault("strides", 1)
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-6))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")

    Z = Conv1D(nbunits, kernel_size=kernel_size,
               strides=strides, padding=padding,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=kernel_regularizer)(X)

    return BatchNormalization(axis=-1)(Z)  # -- CHANNEL_AXIS (-1)


# -----------------------------------------------------------------------
def conv_bn_relu(X, **conv_params):
    Znorm = conv_bn(X, **conv_params)
    return Activation('relu')(Znorm)


# -----------------------------------------------------------------------
def conv_bn_relu_drop(X, **conv_params):
    dropout_rate = conv_params.setdefault("dropout_rate", 0.5)
    A = conv_bn_relu(X, **conv_params)
    return Dropout(dropout_rate)(A)


# -----------------------------------------------------------------------
def conv_bn_relu_spadrop(X, **conv_params):
    dropout_rate = conv_params.setdefault("dropout_rate", 0.5)
    A = conv_bn_relu(X, **conv_params)
    return SpatialDropout1D(dropout_rate)(A)


# -----------------------------------------------------------------------
def conv2d_bn(X, **conv_params):
    nbunits = conv_params["nbunits"];
    kernel_size = conv_params["kernel_size"];

    strides = conv_params.setdefault("strides", 1)
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-6))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")

    Z = Conv2D(nbunits, kernel_size=kernel_size,
               strides=strides, padding=padding,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=kernel_regularizer)(X)

    return BatchNormalization(axis=-1)(Z)  # -- CHANNEL_AXIS (-1)


# -----------------------------------------------------------------------
def conv2d_bn_relu(X, **conv_params):
    Znorm = conv2d_bn(X, **conv_params)
    return Activation('relu')(Znorm)


# -----------------------------------------------------------------------
def conv2d_bn_relu_drop(X, **conv_params):
    dropout_rate = conv_params.setdefault("dropout_rate", 0.5)
    A = conv2d_bn_relu(X, **conv_params)
    return Dropout(dropout_rate)(A)


# -----------------------------------------------------------------------
def conv2d_bn_relu_spadrop(X, **conv_params):
    dropout_rate = conv_params.setdefault("dropout_rate", 0.5)
    A = conv2d_bn_relu(X, **conv_params)
    return SpatialDropout1D(dropout_rate)(A)


# -----------------------------------------------------------------------
def relu_drop(X, **conv_params):
    dropout_rate = conv_params.setdefault("dropout_rate", 0.5)
    A = Activation('relu')(X)
    return Dropout(dropout_rate)(A)


# -----------------------------------------------------------------------
def fc_bn(X, **fc_params):
    nbunits = fc_params["nbunits"];

    kernel_regularizer = fc_params.setdefault("kernel_regularizer", l2(1.e-6))
    kernel_initializer = fc_params.setdefault("kernel_initializer", "he_normal")

    Z = Dense(nbunits, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(X)
    return BatchNormalization(axis=-1)(Z)  # -- CHANNEL_AXIS (-1)


# -----------------------------------------------------------------------
def fc_bn_relu(X, **fc_params):
    Znorm = fc_bn(X, **fc_params)
    return Activation('relu')(Znorm)


# -----------------------------------------------------------------------
def fc_bn_relu_drop(X, **fc_params):
    dropout_rate = fc_params.setdefault("dropout_rate", 0.5)
    A = fc_bn_relu(X, **fc_params)
    return Dropout(dropout_rate)(A)


# -----------------------------------------------------------------------
def softmax(X, nbclasses, **params):
    kernel_regularizer = params.setdefault("kernel_regularizer", l2(1.e-6))
    kernel_initializer = params.setdefault("kernel_initializer", "glorot_uniform")
    return Dense(nbclasses, activation='softmax',
                 kernel_initializer=kernel_initializer,
                 kernel_regularizer=kernel_regularizer)(X)


# -----------------------------------------------------------------------
def getNoClasses(model_path):
    model = load_model(model_path)
    last_weight = model.get_weights()[-1]
    nclasses = last_weight.shape[0]  # --- get size of the bias in the Softmax
    return nclasses


# -----------------------------------------------------------------------
# ---------------------- Training models
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
def trainTestModel(model, Xt_train, y_train, X_test, Y_test_onehot, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.0)

    # ---- optimizer
    opt = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
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
def cv_Model_SISO(model, Xt_train,  ys_train, Xt_val, ys_val, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.01)

    # ---- optimizer
    opt = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

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
def cv_Model_SIMO_st(model, Xt_train,  ys_train, Xt_val, ys_val, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.01)

    # ---- optimizer
    opt = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

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
def cv_Model_SIMO_mt(model, Xt_train,  ys_train, Xt_val, ys_val, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.01)

    # ---- optimizer
    opt = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    # ---- monitoring the minimum validation loss
    checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss',
                                 verbose=0, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    model.compile(optimizer=opt,
                  loss={'out1': 'mse', 'out2': 'mse', 'out3': 'mse'},
                  loss_weights={'out1': 1., 'out2': 1., 'out3': 1.},
                  metrics=['mse'])

    hist = model.fit(Xt_train,
                     {'out1': ys_train[:,[0]], 'out2': ys_train[:,[1]], 'out3': ys_train[:,[2]]},
                     epochs=n_epochs,
                     batch_size=batch_size, shuffle=True,
                     validation_data=(Xt_val,
                                      {'out1': ys_val[:,[0]], 'out2': ys_val[:,[1]], 'out3': ys_val[:,[2]]}),
                     verbose=0, callbacks=callback_list)

    del model
    model = load_model(out_model_file)
    pred = model.predict(x={'ts_input': Xt_val})
    return model, pred

def cv_Model_MIMO(model, Xt_train, Xv_train,  ys_train, Xt_val, Xv_val, ys_val, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.01)

    # ---- optimizer
    opt = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    # ---- monitoring the minimum validation loss
    checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss',
                                 verbose=0, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    model.compile(optimizer=opt,
                  loss={'out1': 'mse', 'out2': 'mse', 'out3': 'mse'},
                  loss_weights={'out1': 1., 'out2': 1., 'out3': 1.},
                  metrics=['mse'])

    hist = model.fit({'ts_input': Xt_train, 'v_input': Xv_train},
                     {'out1': ys_train[:,[0]], 'out2': ys_train[:,[1]], 'out3': ys_train[:,[2]]},
                     epochs=n_epochs,
                     batch_size=batch_size, shuffle=True,
                     validation_data=(Xt_val,
                                      {'out1': ys_val[:,[0]], 'out2': ys_val[:,[1]], 'out3': ys_val[:,[2]]}),
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
    opt = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    # ---- monitoring the minimum validation loss
    checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss',
                                 verbose=0, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    model.compile(optimizer=opt,
                  loss={'out1': 'mse'},
                  loss_weights={'out1': 1.},
                  metrics=['mse'])

    hist = model.fit({'ts_input': Xt_train, 'v_input': Xv_train},
                     {'out1': ys_train},
                     epochs=n_epochs,
                     batch_size=batch_size, shuffle=True,
                     validation_data=({'ts_input': Xt_val, 'v_input': Xv_val},
                                      {'out1': ys_val}),
                     verbose=0, callbacks=callback_list)

    del model
    model = load_model(out_model_file)
    pred = model.predict(x={'ts_input': Xt_val, 'v_input': Xv_val})
    return model, pred


# -----------------------------------------------------------------------
def trainValTestModel_SISO(model, Xt_train,  ys_train, Xt_val, ys_val,
                                Xt_test, ys_test, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.01)

    # ---- optimizer
    opt = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

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
    loss, mse1,  = mdl_ev

    test_time = round(time.time() - start_test_time, 2)
    return model, mse1, hist.history, train_time, test_time


# -----------------------------------------------------------------------
def trainValTestModel_SIMO(model, Xt_train,  ys_train, Xt_val, ys_val,
                                Xt_test, ys_test, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.0)

    # ---- optimizer
    opt = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

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
    opt = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

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
    opt = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
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
