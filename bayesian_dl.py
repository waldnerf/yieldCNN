import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability.python.layers import DenseVariational, DenseReparameterization, DenseFlipout, Convolution2DFlipout, Convolution2DReparameterization
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import *


import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt


print('TensorFlow version:', tf.__version__)
print('TensorFlow Probability version:', tfp.__version__)


def load_dataset(n, w0, b0, x_low, x_high):
    def s(x):
        g = (x - x_low) / (x_high - x_low)
        return 3 * (0.25 + g**2)
    def f(x, w, b):
        return w * x * (1. + np.sin(x)) + b
    x = (x_high - x_low) * np.random.rand(n) + x_low  # N(x_low, x_high)
    x = np.sort(x)
    eps = np.random.randn(n) * s(x)
    y = f(x, w0, b0) + eps
    return x, y

n_data = 500
n_train = 400
w0 = 0.125
b0 = 5.0
x_low, x_high = -20, 60

X, y = load_dataset(n_data, w0, b0, x_low, x_high)
X = np.expand_dims(X, 1)
y = np.expand_dims(y, 1)

idx_randperm = np.random.permutation(n_data)
idx_train = np.sort(idx_randperm[:n_train])
idx_test = np.sort(idx_randperm[n_train:])

X_train, y_train = X[idx_train], y[idx_train]
X_test = X[idx_test]

print("X_train.shape =", X_train.shape)
print("y_train.shape =", y_train.shape)
print("X_test.shape =", X_test.shape)

plt.scatter(X_train, y_train, marker='+', label='Training data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Noisy training data and ground truth')
plt.legend()

def neg_log_likelihood_with_dist(y_true, y_pred):
    return -tf.reduce_mean(y_pred.log_prob(y_true))

batch_size = 100
n_epochs = 3000
lr = 5e-3

def build_point_estimate_model(scale=1):
    model_in = Input(shape=(1,))
    x = Dense(16)(model_in)
    x = LeakyReLU(0.1)(x)
    x = Dense(64)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(16)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(1)(x)
    model_out = DistributionLambda(lambda t: tfd.Normal(loc=t, scale=scale))(x)
    model = Model(model_in, model_out)
    return model

pe_model = build_point_estimate_model()
pe_model.compile(loss=neg_log_likelihood_with_dist, optimizer=Adam(lr), metrics=['mse'])
pe_model.summary()
hist = pe_model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=0)

y_test_pred_pe = pe_model(X_test)
plt.scatter(X_train, y_train, marker='+', label='Training data')
plt.plot(X_test, y_test_pred_pe.mean(), 'r-', marker='+', label='Test data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Noisy training data and ground truth')
plt.legend()



def build_aleatoric_model():
    model_in = Input(shape=(1,))
    x = Dense(16)(model_in)
    x = LeakyReLU(0.1)(x)
    x = Dense(64)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(16)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(tfpl.IndependentNormal.params_size(1))(x)
    model_out = tfpl.IndependentNormal(1)(x)
    model = Model(model_in, model_out)
    return model

al_model = build_aleatoric_model()
al_model.compile(loss=neg_log_likelihood_with_dist, optimizer=Adam(lr), metrics=['mse'])
al_model.summary()
hist = al_model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=0)

y_test_pred_al = al_model(X_test)
y_test_pred_al_mean = y_test_pred_al.mean()
y_test_pred_al_stddev = y_test_pred_al.stddev()
plt.scatter(X_train, y_train, marker='+', label='Training data')
plt.plot(X_test, y_test_pred_al_mean, 'r-', marker='+', label='Test data')
plt.fill_between(np.squeeze(X_test),
                 np.squeeze(y_test_pred_al_mean + 2 * y_test_pred_al_stddev),
                 np.squeeze(y_test_pred_al_mean - 2 * y_test_pred_al_stddev),
                 alpha=0.5, label='Aleatoric uncertainty')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Noisy training data and ground truth')
plt.legend()

#######################################################################################################################

def neg_log_likelihood_with_logits(y_true, y_pred):
    y_pred_dist = tfp.distributions.Categorical(logits=y_pred)
    return -tf.reduce_mean(y_pred_dist.log_prob(tf.argmax(y_true, axis=-1)))

n_class = 10

batch_size = 128
n_epochs = 20
lr = 1e-3

print('Loading MNIST dataset')
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = np.expand_dims(X_train, -1)
n_train = X_train.shape[0]
X_test = np.expand_dims(X_test, -1)
y_train = tf.keras.utils.to_categorical(y_train, n_class)
y_test = tf.keras.utils.to_categorical(y_test, n_class)

# Normalize data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

print("X_train.shape =", X_train.shape)
print("y_train.shape =", y_train.shape)
print("X_test.shape =", X_test.shape)
print("y_test.shape =", y_test.shape)

plt.imshow(X_train[0, :, :, 0], cmap='gist_gray')

def get_kernel_divergence_fn(train_size, w=1.0):
    """
    Get the kernel Kullback-Leibler divergence function

    # Arguments
        train_size (int): size of the training dataset for normalization
        w (float): weight to the function

    # Returns
        kernel_divergence_fn: kernel Kullback-Leibler divergence function
    """
    def kernel_divergence_fn(q, p, _):  # need the third ignorable argument
        kernel_divergence = tfp.distributions.kl_divergence(q, p) / tf.cast(train_size, tf.float32)
        return w * kernel_divergence
    return kernel_divergence_fn

def add_kl_weight(layer, train_size, w_value=1.0):
    w = layer.add_weight(name=layer.name+'/kl_loss_weight', shape=(),
                         initializer=tf.initializers.constant(w_value), trainable=False)
    layer.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)
    return layer


def build_bayesian_bcnn_model(input_shape, train_size):
    model_in = Input(shape=input_shape)
    conv_1 = Convolution2DFlipout(32, kernel_size=(3, 3), padding="same", strides=2,
                                  kernel_divergence_fn=None)
    conv_1 = add_kl_weight(conv_1, train_size)
    x = conv_1(model_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    conv_2 = Convolution2DFlipout(64, kernel_size=(3, 3), padding="same", strides=2,
                                  kernel_divergence_fn=None)
    conv_2 = add_kl_weight(conv_2, train_size)
    x = conv_2(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    dense_1 = DenseFlipout(512, activation='relu',
                           kernel_divergence_fn=None)
    dense_1 = add_kl_weight(dense_1, train_size)
    x = dense_1(x)
    dense_2 = DenseFlipout(10, activation=None,
                           kernel_divergence_fn=None)
    dense_2 = add_kl_weight(dense_2, train_size)
    model_out = dense_2(x)  # logits
    model = Model(model_in, model_out)
    return model


bcnn_model = build_bayesian_bcnn_model(X_train.shape[1:], n_train)
bcnn_model.compile(loss=neg_log_likelihood_with_logits, optimizer=Adam(lr), metrics=['acc'],
                   experimental_run_tf_function=False)
bcnn_model.summary()
hist = bcnn_model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=1, validation_split=0.1)

n_mc_run = 100
med_prob_thres = 0.2

y_pred_logits_list = [bcnn_model.predict(X_test) for _ in range(n_mc_run)]  # a list of predicted logits
y_pred_prob_all = np.concatenate([softmax(y, axis=-1)[:, :, np.newaxis] for y in y_pred_logits_list], axis=-1)
y_pred = [[int(np.median(y) >= med_prob_thres) for y in y_pred_prob] for y_pred_prob in y_pred_prob_all]
y_pred = np.array(y_pred)

idx_valid = [any(y) for y in y_pred]
print('Number of recognizable samples:', sum(idx_valid))

idx_invalid = [not any(y) for y in y_pred]
print('Unrecognizable samples:', np.where(idx_invalid)[0])

print('Test accuracy on MNIST (recognizable samples):',
      sum(np.equal(np.argmax(y_test[idx_valid], axis=-1), np.argmax(y_pred[idx_valid], axis=-1))) / len(y_test[idx_valid]))

print('Test accuracy on MNIST (unrecognizable samples):',
      sum(np.equal(np.argmax(y_test[idx_invalid], axis=-1), np.argmax(y_pred[idx_invalid], axis=-1))) / len(y_test[idx_invalid]))



import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.layers import DenseVariational, DenseReparameterization, DenseFlipout, Convolution2DFlipout, Convolution2DReparameterization
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import *

tfd = tfp.distributions

import numpy as np
import matplotlib.pyplot as plt

def neg_log_likelihood(y_true, y_pred):
    return -y_pred.log_prob(y_true)

def get_neg_log_likelihood_fn(bayesian=False):
    """
    Get the negative log-likelihood function
    # Arguments
        bayesian(bool): Bayesian neural network (True) or point-estimate neural network (False)

    # Returns
        a negative log-likelihood function
    """
    if bayesian:
        def neg_log_likelihood_bayesian(y_true, y_pred):
            labels_distribution = tfp.distributions.Categorical(logits=y_pred)
            log_likelihood = labels_distribution.log_prob(tf.argmax(input=y_true, axis=1))
            loss = -tf.reduce_mean(input_tensor=log_likelihood)
            return loss
        return neg_log_likelihood_bayesian
    else:
        def neg_log_likelihood(y_true, y_pred):
            y_pred_softmax = keras.layers.Activation('softmax')(y_pred)  # logits to softmax
            loss = keras.losses.categorical_crossentropy(y_true, y_pred_softmax)
            return loss
        return neg_log_likelihood


n_class = 10

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = np.expand_dims(X_train, -1)
n_train = X_train.shape[0]
X_test = np.expand_dims(X_test, -1)
n_test = X_test.shape[0]

# Normalize data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

print("X_train.shape =", X_train.shape)
print("y_train.shape =", y_train.shape)
print("X_test.shape =", X_test.shape)
print("y_test.shape =", y_test.shape)

plt.imshow(X_train[0, :, :, 0], cmap='gist_gray')

lr = 1e-3

def build_cnn_model(input_shape):
    model_in = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=3, padding="same", strides=2)(model_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=3, padding="same", strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    model_out = Dense(10, activation='softmax')(x)  # softmax
    model = Model(model_in, model_out)
    return model

def build_bayesian_cnn_model_1(input_shape):
    model_in = Input(shape=input_shape)
    x = Convolution2DFlipout(32, kernel_size=3, padding="same", strides=2)(model_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2DFlipout(64, kernel_size=3, padding="same", strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = DenseFlipout(512, activation='relu')(x)
    model_out = DenseFlipout(10, activation=None)(x)  # logits
    model = Model(model_in, model_out)
    return model

def build_bayesian_cnn_model_2(input_shape):
    model_in = Input(shape=input_shape)
    x = Convolution2DFlipout(32, kernel_size=3, padding="same", strides=2)(model_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2DFlipout(64, kernel_size=3, padding="same", strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = DenseFlipout(512, activation='relu')(x)
    x = DenseFlipout(10, activation=None)(x)  # logits
    model_out = DistributionLambda(lambda t: tfd.Multinomial(logits=t, total_count=1))(x)  # distribution
    model = Model(model_in, model_out)
    return model

cnn_model = build_cnn_model(X_train.shape[1:])
cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
print('CNN Model:')
cnn_model.summary()

bcnn_model_1 = build_bayesian_cnn_model_1(X_train.shape[1:])
bcnn_model_1.compile(loss=get_neg_log_likelihood_fn(bayesian=True), optimizer=Adam(lr), metrics=['accuracy'])
print("BCNN Model 1:")
bcnn_model_1.summary()

bcnn_model_2 = build_bayesian_cnn_model_2(X_train.shape[1:])
bcnn_model_2.compile(loss=neg_log_likelihood, optimizer=Adam(lr), metrics=['accuracy'])
print("BCNN Model 2:")
bcnn_model_2.summary()

batch_size = 128
n_epochs = 5
hist_cnn = cnn_model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=1)
hist_bcnn_1 = bcnn_model_1.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=1)
hist_bcnn_2 = bcnn_model_2.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=1)


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

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.layers import DenseFlipout, Convolution2DFlipout
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Input, \
    GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import *
tfd = tfp.distributions
tfpl = tfp.layers



from outputfiles.plot import *
from outputfiles.save import *
from outputfiles.evaluation import *
from sits.readingsits2D import *
import mysrc.constants as cst
#from deeplearning.architecture_features import *
#from keras.layers import Lambda

fn_indata = cst.my_project.data_dir / f'{cst.target}_full_2d_dataset.pickle'
dir_out = cst.my_project.params_dir

n_channels = 4  # -- NDVI, Rad, Rain, Temp

# ---- Get filenames
print("Input file: ", os.path.basename(str(fn_indata)))

# ---- Downloading
Xt_full, Xv, region_id, groups, y = data_reader(fn_indata)

# ---- Convert region to one hot
region_ohe = add_one_hot(region_id)

# ---- Getting train/val/test data

# ---- variables
n_epochs = 70
batch_size = 800
n_trials = 100

crop_n = 0
month = 3
idx = (month + 1) * 3
Xt = Xt_full[:, :, 0:idx, :]

test_i = np.unique(groups)[0]
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
Xv_test = ohe_test  # np.concatenate([Xv_test[:, [crop_n]], ohe_test], axis=1)

n_batches, image_y, image_x, n_bands = Xt_train.shape

# -- nb_conv CONV layers
input_shape_t = (image_y, image_x, n_bands)
model_in = Input(shape=input_shape_t)
x = keras.layers.convolutional.Conv2D(30, kernel_size=3, padding="same", strides=2)(model_in)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)
x = keras.layers.convolutional.Conv2D(30, kernel_size=3, padding="same", strides=2)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = GlobalMaxPooling2D(data_format='channels_last')(x)
x = Flatten()(x)
x = Dense(10, activation='relu')(x)
model_out_loc = Dense(1, activation='relu')(x)  # logits
model_out_scale = Dense(1, activation='relu')(x)  # logits
model_out = DistributionLambda(lambda t: tfd.Normal(loc=t[0], scale=tf.math.softplus(t[1])))([model_out_loc, model_out_scale])
model = Model(model_in, model_out)

model.summary()

from keras import optimizers
def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)
model.compile(loss=nll,
              optimizer=optimizers.Adam(learning_rate=0.005))

model.fit(Xt_train, ys_train, epochs=20)

y_model = model.predict(Xt_val)



############# THIS WORKS!!!

# -- nb_conv CONV layers
input_shape_t = (image_y, image_x, n_bands)
model_in = Input(shape=input_shape_t)
x = Convolution2DFlipout(30, kernel_size=3, padding="same", strides=2)(model_in)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)
x = Convolution2DFlipout(30, kernel_size=3, padding="same", strides=2)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = GlobalMaxPooling2D(data_format='channels_last')(x)
x = Flatten()(x)
x = DenseFlipout(10, activation='relu')(x)
model_out_loc = DenseFlipout(1, activation='relu')(x)  # logits
model_out_scale = DenseFlipout(1, activation='relu')(x)  # logits
model_out = tfpl.DistributionLambda(lambda t: tfd.Normal(loc=t[0], scale=tf.math.softplus(t[1])))([model_out_loc, model_out_scale])
#x = tfpl.DenseFlipout(tfpl.IndependentNormal.params_size(1), activation='relu')(x)  # logits
#model_out = tfpl.IndependentNormal(1)(x)
model = Model(model_in, model_out)

model.summary()



from keras import optimizers
def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)
model.compile(loss=nll,
              optimizer=optimizers.Adam(learning_rate=0.005))

model.fit(Xt_train, ys_train, epochs=20)

n_preds = 10

y_test_pred_ae_list = [model(Xt_test) for _ in range(n_preds)]
for i, y in enumerate(y_test_pred_ae_list):
    y_mean = y.mean()
    print(y_mean.numpy()[0:5,0])


y_model = model(Xt_val)

tfd = tfp.distributions
tfpl = tfp.layers

divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / Xt_train.shape[0]


input_shape_t = (image_y, image_x, n_bands)
model_in = Input(shape=input_shape_t)
x = tfpl.Convolution2DReparameterization(30, kernel_size=3, padding="same", strides=2,
                                         kernel_prior_fn=tfpl.default_multivariate_normal_fn,
                                         kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                         kernel_divergence_fn=divergence_fn,
                                         bias_prior_fn=tfpl.default_multivariate_normal_fn,
                                         bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                         bias_divergence_fn=divergence_fn
                                         )(model_in)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)
x = tfpl.Convolution2DReparameterization(30, kernel_size=3, padding="same", strides=2,
                                         kernel_prior_fn=tfpl.default_multivariate_normal_fn,
                                         kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                         kernel_divergence_fn=divergence_fn,
                                         bias_prior_fn=tfpl.default_multivariate_normal_fn,
                                         bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                         bias_divergence_fn=divergence_fn
                                         )(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = GlobalMaxPooling2D(data_format='channels_last')(x)
x = Flatten()(x)
x = DenseReparameterization(10, activation='relu',
                            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
                            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                            kernel_divergence_fn=divergence_fn,
                            bias_prior_fn=tfpl.default_multivariate_normal_fn,
                            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                            bias_divergence_fn=divergence_fn
                            )(x)
model_out_loc = DenseReparameterization(1, activation='relu',
                                        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
                                        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                        kernel_divergence_fn=divergence_fn,
                                        bias_prior_fn=tfpl.default_multivariate_normal_fn,
                                        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                        )(x)  # logits
model_out_scale = DenseReparameterization(1, activation='relu',
                                          kernel_prior_fn=tfpl.default_multivariate_normal_fn,
                                          kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                          kernel_divergence_fn=divergence_fn,
                                          bias_prior_fn=tfpl.default_multivariate_normal_fn,
                                          bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                          )(x)  # logits
model_out = DistributionLambda(lambda t: tfd.Normal(loc=t[0], scale=tf.math.softplus(t[1])))([model_out_loc, model_out_scale])
model = Model(model_in, model_out)

model.summary()

model.compile(loss=nll,
              optimizer=optimizers.Adam(learning_rate=0.005))

model.fit(Xt_train, ys_train, epochs=20)

y_model = model.predict(Xt_val)

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

# Create a stochastic encoder -- e.g., for use in a variational auto-encoder.
def normal_sp(params):
  return tfd.Normal(loc=params[..., 0], scale=1e-3 + tf.math.softplus(params[..., 1]))



input_shape_t = (image_y, image_x, 1)
model_in = Input(shape=input_shape_t)
x = Conv2D(32, kernel_size=3, padding="same", strides=2)(model_in)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(32, kernel_size=3, padding="same", strides=2)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Flatten()(x)
x = Dense(60, activation='relu')(x)
x = Dense(2)(x)
model_out = tfpd.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.05 * t[...,1:])))(x)
model = Model(model_in, model_out)



encoder.compile(loss=neg_log_likelihood_with_dist,
              optimizer=optimizers.Adam(learning_rate=0.005),
              metrics=['mse'])

encoder.fit(Xt_train[:,:,:,[0]], ys_train, epochs=20)

y_model = encoder.predict(Xt_val[:,:,:,[0]])


negloglik = lambda y, rv_y: -rv_y.log_prob(y)

w0 = 0.125
b0 = 5.
x_range = [-20, 60]

def load_dataset(n=150, n_tst=150):
  np.random.seed(43)
  def s(x):
    g = (x - x_range[0]) / (x_range[1] - x_range[0])
    return 3 * (0.25 + g**2.)
  x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
  eps = np.random.randn(n) * s(x)
  y = (w0 * x * (1. + np.sin(x)) + b0) + eps
  x = x[..., np.newaxis]
  x_tst = np.linspace(*x_range, num=n_tst).astype(np.float32)
  x_tst = x_tst[..., np.newaxis]
  return y, x, x_tst

y, x, x_tst = load_dataset()
# Build model.
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1),
  tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
])

# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negloglik)
model.fit(x, y, epochs=1000, verbose=False);

# Profit.
[print(np.squeeze(w.numpy())) for w in model.weights];
yhat = model(x_tst)
assert isinstance(yhat, tfd.Distribution)

# Build model.
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1 + 1),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.05 * t[...,1:]))),
])


model_in = Input(shape=(1,))
x = Dense(2)(model_in)
x = Dense(2)(x)
model_out = DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.05 * t[...,1:])))(x)
model = Model(model_in, model_out)



model_in = Input(shape=input_shape_t)
x = Conv2D(32, kernel_size=3, padding="same", strides=2)(model_in)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(32, kernel_size=3, padding="same", strides=2)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Flatten()(x)
x = Dense(60, activation='relu')(x)
x = Dense(2)(x)
model_out = DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.05 * t[...,1:])))(x)
model = Model(model_in, model_out)

model.summary()


kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (Xt_train.shape[0] *1.0)

model_vi = Sequential()
#model_vi.add(keras.layers.InputLayer(input_shape=input_shape_t))
#model_vi.add(tfp.layers.Convolution2DFlipout(8,kernel_size=(3,3),padding="same", activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
#model_vi.add(tf.keras.layers.MaxPooling2D((2,2)))
#model_vi.add(tfp.layers.Convolution2DFlipout(16,kernel_size=(3,3),padding="same", activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
#model_vi.add(tf.keras.layers.MaxPooling2D((2,2)))
model_vi.add(tf.keras.layers.Flatten(input_shape=input_shape_t))
#model_vi.add(tfp.layers.DenseFlipout(100, activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
#model_vi.add(tfp.layers.DenseFlipout(2, activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
model_vi.add(Dense(100, activation = 'relu'))
model_vi.add(Dense(2, activation = 'relu'))
#model_vi.add(DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.05 * t[...,1:]))))
model_vi.add(tfp.layers.IndependentNormal(1))
#x = Dense(tfpl.IndependentNormal.params_size(1))(x)
#model_out = tfpl.IndependentNormal(1)(x)

#model_vi.compile(loss=, optimizer="adam", metrics=['accuracy'])
model_vi.summary()

model_vi.compile(loss=nll,
              optimizer=optimizers.Adam(learning_rate=0.005))

model_vi.fit(Xt_train, ys_train, epochs=20)

y_model = model_vi.predict(Xt_val)


model = Sequential()
model.add(Conv2D(8,kernel_size=(3,3),padding="same", activation = 'relu',input_shape=input_shape_t))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(Conv2D(8,kernel_size=(3,3),padding="same", activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Flatten())
model.add(Dense(100, activation = 'relu'))
model.add(Dense(2, activation = 'relu'))
model.add(DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.05 * t[...,1:]))))

model.compile(loss=nll, optimizer=optimizers.Adam(learning_rate=0.005))

#model_vi.compile(loss=, optimizer="adam", metrics=['accuracy'])
model.summary()


model.fit(Xt_train, ys_train, epochs=20)

y_model = model.predict(Xt_val)

###
model_in = Input(shape=input_shape_t)
x = Flatten()(model_in)
x = Dense(64)(x)
x = LeakyReLU(0.1)(x)
x = Dense(64)(x)
x = LeakyReLU(0.1)(x)
x = Dense(16)(x)
x = LeakyReLU(0.1)(x)
model_out_loc = Dense(1)(x)
model_out_scale = Dense(1)(x)
model_out = DistributionLambda(lambda t: tfd.Normal(loc=t[0],
                                                    scale=1e-7 + tf.math.softplus(1e-3 * t[1])))([model_out_loc,
                                                                                                  model_out_scale])
model = Model(model_in, model_out)


model.compile(loss=nll, optimizer=optimizers.Adam(learning_rate=0.005))

#model_vi.compile(loss=, optimizer="adam", metrics=['accuracy'])
model.summary()


model.fit(Xt_train, ys_train, epochs=20)

y_model = model(Xt_val)