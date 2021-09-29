#!/usr/bin/python

"""
Defining keras architecture.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.layers import DenseFlipout, Convolution2DFlipout
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Input, \
    GlobalMaxPooling2D, MaxPooling2D, Concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import *
from keras.callbacks import Callback, ModelCheckpoint, History, EarlyStopping
from keras.models import load_model

tfd = tfp.distributions
tfpl = tfp.layers

from keras.regularizers import l2

import numpy as np

# -----------------------------------------------------------------------
# ---------------------- ARCHITECTURES
# ------------------------------------------------------------------------

# -----------------------------------------------------------------------

def Archi_prob_2DCNNw_MISO(Xt, Xv, nbunits_conv=10, kernel_size=3, strides=3, pool_size=3, nb_fc=1,
                           funits_fc=1, activation='sigmoid', v_fc=1, nbunits_v=10, verbose=True):
    # -- get the input sizes
    if isinstance(Xt, list):
        input_shape_t = (Xt[0], Xt[1])
        input_shape_v = (Xv[0],)
    else:
        n_batches, image_y, image_x, n_bands = Xt.shape
        input_shape_t = (image_y, image_x, n_bands)
        mv, Lv = Xv.shape
        input_shape_v = (Lv,)

    # Define the input placeholder.
    xt_input = Input(input_shape_t, name='ts_input')
    xv_input = Input(input_shape_v, name='v_input')

    # -- nb_conv CONV layers
    xt = Convolution2DFlipout(nbunits_conv, kernel_size=kernel_size, padding="same", strides=1)(xt_input)
    xt = BatchNormalization()(xt)
    xt = Activation(activation='relu')(xt)
    xt = MaxPooling2D(pool_size=pool_size, strides=strides, padding='valid')(xt)
    xt = Convolution2DFlipout(nbunits_conv, kernel_size=kernel_size, padding="same", strides=1)(xt)
    xt = BatchNormalization()(xt)
    xt = Activation(activation)(xt)
    xt = GlobalMaxPooling2D(data_format='channels_last')(xt)

    # -- Flatten
    xt = Flatten()(xt)

    # -- Vector inputs
    if v_fc == 1:
       xv = DenseFlipout(nbunits_v, activation=activation)(xv_input)
    else:
        xv = xv_input

    # -- Concatenate
    x = Concatenate()([xt, xv])

    # -- Output FC layers
    for add in range(nb_fc - 1):
       x = DenseFlipout(nbunits_conv * funits_fc, activation=activation)(x)

    #model_out_loc = DenseFlipout(1, activation='relu')(x)  # logits
    #model_out_scale = DenseFlipout(1)(x)  # logits
    #model_out = DistributionLambda(lambda t: tfd.Normal(loc=t[0], scale=tf.math.softplus(t[1])))(
    #    [model_out_loc, model_out_scale])
    model_out = DenseFlipout(1, activation='relu')(x)  # logits

    # Create model.
    model = Model(inputs=[xt_input, xv_input], outputs=model_out, name=f'Archi_prob_CNN_MISO')
    #model = Model(inputs=xt_input, outputs=model_out, name=f'Archi_prob_CNN_MISO')
    if verbose:
        model.summary()
    return model

def Archi_prob_2DCNNw_MISO(Xt, Xv, nbunits_conv=10, kernel_size=3, strides=3, pool_size=3, nb_fc=1,
                           funits_fc=1, activation='sigmoid', v_fc=1, nbunits_v=10, verbose=True):
    # -- get the input sizes
    if isinstance(Xt, list):
        input_shape_t = (Xt[0], Xt[1])
        input_shape_v = (Xv[0],)
    else:
        n_batches, image_y, image_x, n_bands = Xt.shape
        input_shape_t = (image_y, image_x, n_bands)
        mv, Lv = Xv.shape
        input_shape_v = (Lv,)

    # Define the input placeholder.
    xt_input = Input(input_shape_t, name='ts_input')
    xv_input = Input(input_shape_v, name='v_input')

    # -- nb_conv CONV layers
    xt = Convolution2DFlipout(nbunits_conv, kernel_size=kernel_size, padding="same", strides=1)(xt_input)
    xt = BatchNormalization()(xt)
    xt = Activation(activation)(xt)
    xt = MaxPooling2D(pool_size=pool_size, strides=strides, padding='valid')(xt)
    xt = Convolution2DFlipout(nbunits_conv, kernel_size=kernel_size, padding="same", strides=1)(xt)
    xt = BatchNormalization()(xt)
    xt = Activation(activation)(xt)
    xt = GlobalMaxPooling2D(data_format='channels_last')(xt)

    # -- Flatten
    xt = Flatten()(xt)

    # -- Vector inputs
    if v_fc == 1:
       xv = DenseFlipout(nbunits_v, activation=activation)(xv_input)
    else:
        xv = xv_input

    # -- Concatenate
    x = Concatenate()([xt, xv])

    # -- Output FC layers
    for add in range(nb_fc - 1):
       x = DenseFlipout(nbunits_conv * funits_fc, activation=activation)(x)

    #model_out_loc = DenseFlipout(1, activation='relu')(x)  # logits
    #model_out_scale = DenseFlipout(1)(x)  # logits
    #model_out = DistributionLambda(lambda t: tfd.Normal(loc=t[0], scale=tf.math.softplus(t[1])))(
    #    [model_out_loc, model_out_scale])
    model_out = DenseFlipout(1, activation='relu')(x)  # logits

    # Create model.
    model = Model(inputs=[xt_input, xv_input], outputs=model_out, name=f'Archi_prob_CNN_MISO')
    #model = Model(inputs=xt_input, outputs=model_out, name=f'Archi_prob_CNN_MISO')
    if verbose:
        model.summary()
    return model

def Archi_prob3_2DCNNw_MISO(Xt, Xv, nbunits_conv=10, kernel_size=3, strides=3, pool_size=3, nb_fc=1,
                           funits_fc=1, activation='sigmoid', v_fc=1, nbunits_v=10, verbose=True):
    # -- get the input sizes
    if isinstance(Xt, list):
        input_shape_t = (Xt[0], Xt[1])
        input_shape_v = (Xv[0],)
    else:
        n_batches, image_y, image_x, n_bands = Xt.shape
        input_shape_t = (image_y, image_x, n_bands)
        mv, Lv = Xv.shape
        input_shape_v = (Lv,)

    # Define the input placeholder.
    xt_input = Input(input_shape_t, name='ts_input')
    xv_input = Input(input_shape_v, name='v_input')

    # -- nb_conv CONV layers
    xt = Conv2D(nbunits_conv, kernel_size=kernel_size, padding="same", strides=1, kernel_initializer="he_normal",
               kernel_regularizer=l2(1.e-6))(xt_input)
    xt = BatchNormalization(axis=-1)(xt)
    xt = Activation(activation='relu')(xt)
    xt = MaxPooling2D(pool_size=pool_size, strides=strides, padding='valid')(xt)
    xt = Conv2D(nbunits_conv, kernel_size=kernel_size, padding="same", strides=1, kernel_initializer="he_normal",
               kernel_regularizer=l2(1.e-6))(xt)
    xt = BatchNormalization(axis=-1)(xt)
    xt = Activation(activation='relu')(xt)
    xt = GlobalMaxPooling2D(data_format='channels_last')(xt)

    # -- Flatten
    xt = Flatten()(xt)

    # -- Vector inputs
    if v_fc == 1:
       xv = Dense(nbunits_v, activation=activation)(xv_input)
    else:
        xv = xv_input

    # -- Concatenate
    x = Concatenate()([xt, xv])

    # -- Output FC layers
    for add in range(nb_fc - 1):
       x = Dense(nbunits_conv * funits_fc, activation=activation)(x)

    model_out_loc = Dense(1)(x)  # logits
    model_out_scale = Dense(1)(x)  # logits
    model_out = DistributionLambda(lambda t: tfd.Normal(loc=t[0], scale=tf.math.softplus(t[1])))(
        [model_out_loc, model_out_scale])


    # Create model.
    model = Model(inputs=[xt_input, xv_input], outputs=model_out, name=f'Archi_prob_CNN_MISO')
    #model = Model(inputs=xt_input, outputs=model_out, name=f'Archi_prob_CNN_MISO')
    if verbose:
        model.summary()
    return model

def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


def inference_total_uncertainty(model, X_inference, scaler_=None, n_model_preds=20, n_draws=100):
    if isinstance(X_inference, dict):
        n_smpl_batch = X_inference[list(X_inference)[0]].shape[0]
    else:
        n_smpl_batch = X_inference.shape[0]
    #TODO: change this
    n_model_preds = 1#n_draws = 1
    preds_ = np.zeros((n_model_preds, n_draws, n_smpl_batch))

    y_test_pred_ae_list = [model(X_inference) for _ in range(n_model_preds)]
    for i, y in enumerate(y_test_pred_ae_list):
        y_preds = y.sample(100) #TODO
        if scaler_ is None:
            preds_[i, :, :] = y_preds[:, :, 0] #TODO: y_preds[:, :, 0]
        else:
            preds_[i, :, :] = scaler_.inverse_transform(y_preds[:, :, 0]) # TODO: scaler_.inverse_transform(y_preds[:, :, 0])
    preds_mean_ = y.mean().numpy()  #preds_.mean(axis=(0, 1))
    preds_std_ = y.stddev().numpy()  #preds_.std(axis=(0, 1))

    #return np.expand_dims(preds_mean_, axis=1), np.expand_dims(preds_std_, axis=1)
    return preds_mean_, preds_std_

def cv_Model_MISO(model, Xt_train, Xv_train, ys_train, Xt_val, Xv_val, ys_val, out_model_file, **train_params):
    # ---- variables
    n_epochs = train_params.setdefault("n_epochs", 20)
    batch_size = train_params.setdefault("batch_size", 32)

    lr = train_params.setdefault("lr", 0.001)
    beta_1 = train_params.setdefault("beta_1", 0.9)
    beta_2 = train_params.setdefault("beta_2", 0.999)
    decay = train_params.setdefault("decay", 0.01)

    # ---- optimizer
    opt = Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    # ---- monitoring the minimum validation loss
    checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss',
                                 verbose=0, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    model.compile(optimizer=opt,
                  loss='mse', #todo
                  metrics=['mse'])

    model_hist = model.fit({'ts_input': Xt_train, 'v_input': Xv_train},
                           ys_train,
                           epochs=n_epochs,
                           batch_size=batch_size,
                           shuffle=True,
                           validation_data=({'ts_input': Xt_val, 'v_input': Xv_val}, ys_val),
                           verbose=0, callbacks=callback_list)

    model.load_weights(out_model_file)

    X_inference = {'ts_input': Xt_val, 'v_input': Xv_val}
    y_means, y_stds = inference_total_uncertainty(model, X_inference)

    return model, y_means, y_stds

import matplotlib.pyplot as plt
def plot_training_history(model_hist):
    plt.plot(model_hist.history['mse'])
    plt.plot(model_hist.history['val_mse'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(model_hist.history['loss'])
    plt.plot(model_hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()