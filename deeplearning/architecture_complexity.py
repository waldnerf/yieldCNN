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
from keras.layers import GRU, Bidirectional, LSTM
from keras import backend as K

#-----------------------------------------------------------------------
#---------------------- ARCHITECTURES
#------------------------------------------------------------------------	

#-----------------------------------------------------------------------
def Archi_CONV_SIMO(Xt, nb_conv=2, nbunits_conv=8, nbunits_fc=256, verbose=True):
	
	#-- get the input sizes
	mt, Lt, deptht = Xt.shape
	input_shape_t = (Lt, deptht)
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5

	# Define the input placeholder.
	Xt_input = Input(input_shape_t, name='ts_input')

	#-- nb_conv CONV layers
	cnt = 1
	Xt = Xt_input
	Xt = conv_bn_relu_drop(Xt, nbunits=nbunits_conv, kernel_size=1, kernel_regularizer=l2(l2_rate),
						   dropout_rate=dropout_rate)
	for add in range(nb_conv):
		Xt = conv_bn_relu_drop(Xt, nbunits=nbunits_conv*cnt, kernel_size=5, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		Xt = MaxPooling1D(pool_size=2, strides=2, padding='valid')(Xt)
		cnt += 1

	#-- Flatten + 	2 FC layers
	X = Flatten()(Xt)
	X = fc_bn(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	X = Dense(nbunits_fc/2)(X)
	out1 = Dense(1, activation='linear', name='out1')(X)
	out2 = Dense(1, activation='linear', name='out2')(X)
	out3 = Dense(1, activation='linear', name='out3')(X)

	# Create model.
	#model = Model(inputs=[Xt_input, Xv_input], outputs=[out1, out2, out3], name=f'Archi_{nb_conv}CONV_FC')
	model = Model(inputs=Xt_input, outputs=[out1, out2, out3], name=f'Archi_{nb_conv}CONV_FC')
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


# -----------------------------------------------------------------------
def Archi_RNN_FC(Xt, Xv, nb_rnn=1, nbunits_rnn=16, nbunits_fc=32, verbose=True):

	# -- get the input sizes
	mt, Lt, deptht = Xt.shape
	input_shape_t = (Lt, deptht)
	mv, Lv = Xv.shape
	input_shape_v = (Lv,)
	# -- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.25

	# Define the input placeholder.
	Xt_input = Input(input_shape_t, name='ts_input')
	Xv_input = Input(input_shape_v, name='v_input')

	# -- nb_conv CONV layers
	Xt = Xt_input
	for add in range(nb_rnn - 1):
		Xt = Bidirectional(LSTM(nbunits_rnn//2, return_sequences=True))(Xt)
		Xt = Dropout(dropout_rate)(Xt)
	Xt = Bidirectional(LSTM(nbunits_rnn//2, return_sequences=False))(Xt)
	Xt = Dropout(dropout_rate)(Xt)

	Xt = Flatten()(Xt)
	X = layers.Concatenate()([Xt, Xv_input])
	# -- 1 FC layers
	X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	X = Dense(nbunits_fc)(X)
	# -- Output layer
	out1 = Dense(1, activation='linear', name='out1')(X)

	# Create model.
	model = Model(inputs=[Xt_input, Xv_input], outputs=out1, name='Archi_RNN_1FC')
	if verbose:
		model.summary()
	return model

# -----------------------------------------------------------------------
def Archi_RNN_FC_MIMO(Xt, Xv, nb_rnn=1, nbunits_rnn=16, nbunits_fc=32, verbose=True):

	# -- get the input sizes
	mt, Lt, deptht = Xt.shape
	input_shape_t = (Lt, deptht)
	mv, Lv = Xv.shape
	input_shape_v = (Lv,)
	# -- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.25

	# Define the input placeholder.
	Xt_input = Input(input_shape_t, name='ts_input')
	Xv_input = Input(input_shape_v, name='v_input')

	# -- nb_conv CONV layers
	Xt = Xt_input
	for add in range(nb_rnn - 1):
		Xt = LSTM(nbunits_rnn, return_sequences=True)(Xt)
		Xt = Dropout(dropout_rate)(Xt)
	Xt = LSTM(nbunits_rnn, return_sequences=False)(Xt)
	Xt = Dropout(dropout_rate)(Xt)

	Xt = Flatten()(Xt)
	X = layers.Concatenate()([Xt, Xv_input])
	# -- 1 FC layers
	X = fc_bn(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate))
	#X = Dense(nbunits_fc)(X)
	# -- Output layer
	out1 = Dense(1, activation='linear', name='out1')(X)
	out2 = Dense(1, activation='linear', name='out2')(X)
	out3 = Dense(1, activation='linear', name='out3')(X)

	# Create model.
	model = Model(inputs=[Xt_input, Xv_input], outputs=[out1, out2, out3], name='Archi_RNN_MIMO')
	if verbose:
		model.summary()
	return model


# -----------------------------------------------------------------------
def Archi_RNN_FC_SIMO(Xt,  nb_rnn=1, nbunits_rnn=16, nbunits_fc=32, verbose=True):

	# -- get the input sizes
	mt, Lt, deptht = Xt.shape
	input_shape_t = (Lt, deptht)
	# -- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.25

	# Define the input placeholder.
	Xt_input = Input(input_shape_t, name='ts_input')

	# -- nb_conv CONV layers
	X = Xt_input
	for add in range(nb_rnn - 1):
		X = LSTM(nbunits_rnn, return_sequences=True)(X)
		X = Dropout(dropout_rate)(X)
	X = LSTM(nbunits_rnn, return_sequences=False)(X)
	X = Dropout(dropout_rate)(X)

	X = Flatten()(X)
	# -- 1 FC layers
	X = fc_bn(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate))
	#X = Dense(nbunits_fc)(X)
	# -- Output layer
	out1 = Dense(1, activation='linear', name='out1')(X)
	out2 = Dense(1, activation='linear', name='out2')(X)
	out3 = Dense(1, activation='linear', name='out3')(X)

	# Create model.
	model = Model(inputs=Xt_input, outputs=[out1, out2, out3], name='Archi_RNN_SIMO')
	if verbose:
		model.summary()
	return model


#EOF
