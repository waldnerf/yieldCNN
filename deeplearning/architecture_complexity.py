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
from keras import backend as K

#-----------------------------------------------------------------------
#---------------------- ARCHITECTURES
#------------------------------------------------------------------------	

#-----------------------------------------------------------------------
def Archi_3CONV_2FC(X, nbunits_conv=64, nbunits_fc=256):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3

	# Define the input placeholder.
	X_input = Input(input_shape, name='cnn_input')

	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	out1 = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	out1 = Dense(1, activation='relu', name='age_output')(out1)
	out2 = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	out2 = Dense(1, activation='relu', name='age_output')(out2)
		
	# Create model.
	return Model(inputs=X_input, outputs=[out1, out2], name='Archi_3CONV_2FC')

#--------------------- Switcher for running the architectures
def runArchi(noarchi, *args):
	#---- variables
	n_epochs = 20
	batch_size = 32
	
	switcher = {		
		0: Archi_3CONV_2FC
	}
	func = switcher.get(noarchi, lambda: 0)
	model = func(args[0], args[1].shape[1])
	
	if len(args)==5:
		return trainTestModel_EarlyAbandon(model, *args, n_epochs=n_epochs, batch_size=batch_size)
	elif len(args)==7:
		return trainValTestModel_EarlyAbandon(model, *args, n_epochs=n_epochs, batch_size=batch_size)

#EOF
