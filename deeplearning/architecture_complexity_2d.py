#!/usr/bin/python

""" 
	Defining keras architecture.
	4.4. How big and deep model for our data?
	4.4.1. Width influence or the bias-variance trade-off

	Multiple inputs: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
	Multiple outputs: https://github.com/rahul-pande/faces-mtl/blob/master/faces_mtl_age_gender.ipynb
"""

import sys, os
from deeplearning.architecture_features import *
import keras
from keras import layers
from keras.layers import Flatten
from keras.layers import GRU, Bidirectional, LSTM, GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras import backend as K

# -----------------------------------------------------------------------
# ---------------------- ARCHITECTURES
# ------------------------------------------------------------------------

# -----------------------------------------------------------------------
def Archi_2DCNNw_MISO(Xt, Xv, nbunits_conv=10, kernel_size=3, strides=3, pool_size=3, dropout_rate=0., nb_fc=1,
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
    # -- parameters of the architecture
    l2_rate = 1.e-6

    # Define the input placeholder.
    Xt_input = Input(input_shape_t, name='ts_input')
    Xv_input = Input(input_shape_v, name='v_input')

    # -- nb_conv CONV layers
    Xt = Xt_input
    Xt = conv2d_bn_relu_drop(Xt, nbunits=nbunits_conv, kernel_size=kernel_size, dropout_rate=dropout_rate,
                             kernel_regularizer=l2(l2_rate))
    Xt = MaxPooling2D(pool_size=pool_size, strides=strides, padding='valid')(Xt)
    Xt = conv2d_bn_relu_drop(Xt, nbunits=nbunits_conv, kernel_size=kernel_size, dropout_rate=dropout_rate,
                             kernel_regularizer=l2(l2_rate))
    Xt = GlobalMaxPooling2D(data_format='channels_last')(Xt)

    # -- Flatten
    Xt = Flatten()(Xt)

    # -- Vector inputs
    Xv = Xv_input
    if v_fc == 1:
        Xv = Dense(nbunits_v, activation=activation)(Xv)

    # -- Concatenate
    X = layers.Concatenate()([Xt, Xv])

    # -- Output FC layers
    for add in range(nb_fc - 1):
        X = Dense(nbunits_conv * funits_fc, activation=activation)(X)
        X = Dropout(dropout_rate)(X)

    out1 = Dense(1, activation='relu', name='out1')(X)

    # Create model.
    model = Model(inputs=[Xt_input, Xv_input], outputs=[out1], name=f'Archi_CNNw_MISO')
    if verbose:
        model.summary()
    return model



def Archi_CONV_MIMO(Xt, Xv, nb_conv=2, nbunits_conv=8, nbunits_fc=256, verbose=True):
    # -- get the input sizes
    mt, Lt, deptht = Xt.shape
    input_shape_t = (Lt, deptht)
    mv, Lv = Xv.shape
    input_shape_v = (Lv,)
    # -- parameters of the architecture
    l2_rate = 1.e-6
    dropout_rate = 0.5

    # Define the input placeholder.
    Xt_input = Input(input_shape_t, name='ts_input')
    Xv_input = Input(input_shape_v, name='v_input')

    # -- nb_conv CONV layers
    cnt = 1
    Xt = Xt_input
    Xt = conv_bn_relu_drop(Xt, nbunits=nbunits_conv, kernel_size=1, kernel_regularizer=l2(l2_rate),
                           dropout_rate=dropout_rate)
    for add in range(nb_conv):
        Xt = conv_bn_relu_drop(Xt, nbunits=nbunits_conv * cnt, kernel_size=5, kernel_regularizer=l2(l2_rate),
                               dropout_rate=dropout_rate)
        Xt = MaxPooling1D(pool_size=2, strides=2, padding='valid')(Xt)
        cnt += 1

    # -- Flatten + 	2 FC layers
    Xt = Flatten()(Xt)
    X = layers.Concatenate()([Xt, Xv_input])
    X = fc_bn(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
    X = Dense(nbunits_fc / 2)(X)
    out1 = Dense(1, activation='linear', name='out1')(X)
    out2 = Dense(1, activation='linear', name='out2')(X)
    out3 = Dense(1, activation='linear', name='out3')(X)

    # Create model.
    model = Model(inputs=[Xt_input, Xv_input], outputs=[out1, out2, out3], name=f'Archi_{nb_conv}CONV_FC')
    if verbose:
        model.summary()
    return model


