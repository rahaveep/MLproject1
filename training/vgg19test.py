# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

###########
# IMPORTS #
###########

import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers
import time

import tensorflow as tf
from keras.models import Model
#print(tf.__version__)

from pickle import dump
'''    
from tensorflow.keras.datasets import cifar10   # MNIST dataset is included in Keras
from tensorflow.keras.models import Sequential  # Model type to be used

from tensorflow.keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from tensorflow.keras.utils import np_utils                         # NumPy related tools
'''
###############
# IMPORT DATA #
###############

# import CIFAR10, later import data for project
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# visualize random samples of the data #
plt.rcParams['figure.figsize'] = (4,4) # Make the figures a bit bigger

for i in range(9):
    plt.subplot(3,3,i+1)
    num = random.randint(0, len(x_test))
    plt.imshow(x_test[num], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_test[num]))
    
plt.tight_layout()


################
# FORMAT INPUT #
################


# format input

'''#
X_train = x_train.reshape(50000, 3072) # reshape 45,000 32x32 matrices into 45,000 3072-length vectors.
X_test = x_test.reshape(10000, 3072)   # reshape 15,000 32 x 32 matrices into 15,000 3072-length vectors.

X_train = X_train.astype('float32')    # change integers to 32-bit floating point numbers
X_test = X_test.astype('float32')
'''
X_train = x_train.astype('float32')    # change integers to 32-bit floating point numbers
X_test = x_test.astype('float32')
#'''#

X_train /= 255                         # normalize each value for each pixel for the entire vector for each input
X_test /= 255
print("Input training matrix shape", X_train.shape)
print("Input testing matrix shape", X_test.shape)

# format output

nb_classes = 10 # number of unique digits
from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

print("Output training matrix shape", Y_train.shape)
print("Output testing matrix shape", Y_test.shape, '\n')


################
# IMPORT MODEL #
################

# model documentation link: https://keras.io/api/applications/vgg/
'''
model = tf.keras.applications.vgg19.VGG19(
    include_top=True, 
    weights='imagenet', 
    input_tensor=None, 
    input_shape=X_train.shape[1:])
'''
model = tf.keras.applications.VGG19(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=X_train.shape[1:],
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

# add new classifier layers
#tf.keras.layers.Flatten(data_format=None, **kwargs)
flat1 = tf.keras.layers.Flatten()(model.layers[-1].output)
class1 = tf.keras.layers.Dense(256, activation='relu')(flat1)
output = tf.keras.layers.Dense(10, activation='softmax')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)

model.summary()

###############
# TRAIN MODEL #
###############

model.compile(
    loss='mean_squared_error',
    optimizer='Adam',
    metrics=['accuracy'])

model.fit(
    X_train, 
    Y_train,      
    batch_size=256, 
    epochs=100,
    verbose=1)


model.save('my_model.h5')