#!/usr/bin/python

""" 
Defining tensorflow.keras architecture, and training the models
"""

from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Flatten, Lambda, \
    SpatialDropout1D, \
    Concatenate
from tensorflow.keras.layers import Conv1D, Conv2D, AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D, \
    GlobalAveragePooling1D

# -----------------------------------------------------------------------
# ---------------------- Modules
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# MM test batch after ferlu
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

    return Activation('relu')(Z)
# -----------------------------------------------------------------------
def conv_bn_relu_norm(X, **conv_params):
    Z = conv_bn_relu(X, **conv_params)
    return BatchNormalization(axis=-1)(Z)  # -- CHANNEL_AXIS (-1)
# -----------------------------------------------------------------------
def conv_bn_relu_norm_drop(X, **conv_params):
    dropout_rate = conv_params.setdefault("dropout_rate", 0.5)
    A = conv_bn_relu_norm(X, **conv_params)
    return Dropout(dropout_rate)(A)



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
def flipout_conv2d_bn(X, **conv_params):
    nbunits = conv_params["nbunits"];
    kernel_size = conv_params["kernel_size"];

    strides = conv_params.setdefault("strides", 1)
    padding = conv_params.setdefault("padding", "same")

    Z = tfp.layers.Convolution2DFlipout(nbunits, kernel_size=kernel_size, strides=strides, padding=padding)(X)

    return BatchNormalization(axis=-1)(Z)  # -- CHANNEL_AXIS (-1)


# -----------------------------------------------------------------------
def flipout_conv2d_bn_relu(X, **conv_params):
    Znorm = flipout_conv2d_bn(X, **conv_params)
    return Activation('relu')(Znorm)


# -----------------------------------------------------------------------
def flipout_conv2d_bn_relu_drop(X, **conv_params):
    dropout_rate = conv_params.setdefault("dropout_rate", 0.5)
    A = flipout_conv2d_bn_relu(X, **conv_params)
    return Dropout(dropout_rate)(A)

# EOF
