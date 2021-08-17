import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Activation, MaxPooling2D, Dense
from tensorflow_addons.layers import SpatialPyramidPooling2D

batch_size = 64
num_channels = 3
num_classes = 10

Xt_input = Input((64, 64, 3), name='ts_input')
X = Convolution2D(filters=6, kernel_size=3)(Xt_input)
X = Activation('relu')(X)
X = Convolution2D(filters=6, kernel_size=3)(Xt_input)
X = Activation('relu')(X)
X = Convolution2D(filters=6, kernel_size=3)(Xt_input)
X = Activation('relu')(X)
X = SpatialPyramidPooling2D([1, 2])(X)
X = Flatten()(X)
out1 = Dense(1, activation='relu', name='out1')(X)

model = Model(inputs=[Xt_input], outputs=[out1], name=f'Archi_CNNw_MISO')
model.summary()


# uses theano ordering. Note that we leave the image size as None to allow multiple image sizes
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, None, None)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(SpatialPyramidPooling([1, 2, 4]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd')

# train on 64x64x3 images
model.fit(np.random.rand(batch_size, num_channels, 64, 64), np.zeros((batch_size, num_classes)))
# train on 32x32x3 images