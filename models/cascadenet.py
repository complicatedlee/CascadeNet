
# -*- coding: utf-8 -*-
'''CascadeNet model for Keras.
# Reference:
- [CascadeNet: Modified ResNet with Cascade Blocks]
'''
from __future__ import print_function

import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Activation, Add, BatchNormalization, advanced_activations, Cropping2D
from keras.layers import Conv2D, ZeroPadding2D, UpSampling2D
from keras.layers import MaxPooling2D, Dropout, AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import concatenate
from keras import backend as K
from keras.engine.topology import get_source_inputs

from custom_layers.scale_layer import Scale
from keras import regularizers

# ===================cascade net=============
__all__ = ['net18', 'net34', 'net50', 'net101', 'net20', 'net32', 'net44', 'net56', 'net110']

global bn_axis, eps
bn_axis = 3
eps = 1.1e-5
weight_decay = 0.0001

def block(input_tensor, kernel_size, filters, strides = (2, 2), shortcut = None):
    filters1, filters2 = filters

    x = Conv2D(filters1, (3, 3), subsample=strides, padding="same",use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis)(x)
    x = Scale(axis=bn_axis)(x)
    x = Activation('relu')(x)#advanced_activations.LeakyReLU(alpha=0.2)(x)#

    if shortcut == None:
        shortcut = x
    else:
        if strides == (2, 2):
            input_tensor = Conv2D(filters2, (1, 1), subsample=strides, padding="same",use_bias=False)(input_tensor)
            input_tensor = BatchNormalization(epsilon=eps, axis=bn_axis)(input_tensor)
            input_tensor = Activation('relu')(input_tensor)#advanced_activations.LeakyReLU(alpha=0.2)(input_tensor)#

            shortcut = Conv2D(filters1, (1, 1), subsample=strides, padding="same", use_bias=False)(shortcut)
            shortcut = BatchNormalization(epsilon=eps, axis=bn_axis)(shortcut)
            shortcut = Activation('relu')(shortcut)#advanced_activations.LeakyReLU(alpha=0.2)(shortcut)#

        shortcut = concatenate([x, shortcut])

    x = Conv2D(filters2, (kernel_size, kernel_size),padding="same", use_bias=False)(shortcut)
    x = BatchNormalization(epsilon=eps, axis=bn_axis)(x)
    x = Scale(axis=bn_axis)(x)
    x = Activation('relu')(x)#advanced_activations.LeakyReLU(alpha=0.2)(x)#
 
    x = concatenate([x, input_tensor])
    return x, shortcut

def b_block(input_tensor, kernel_size, filters, strides = (2, 2), shortcut = None):
    filters1, filters2 = filters

    x = Conv2D(filters1, (1, 1), subsample=strides, padding="same",use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis)(x)
    x = Scale(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, (kernel_size, kernel_size), padding="same", use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis)(x)
    x = Scale(axis=bn_axis)(x)
    x = Activation('relu')(x)

    if shortcut == None:
        shortcut = x
    else:
        if strides == (2, 2):
            input_tensor = Conv2D(filters2, (1, 1), subsample=strides, padding="same",use_bias=False)(input_tensor)
            input_tensor = BatchNormalization(epsilon=eps, axis=bn_axis)(input_tensor)
            input_tensor = Activation('relu')(input_tensor)

            shortcut = Conv2D(filters1, (1, 1), subsample=strides, padding="same", use_bias=False)(shortcut)
            shortcut = BatchNormalization(epsilon=eps, axis=bn_axis)(shortcut)
            shortcut = Activation('relu')(shortcut)

        shortcut = concatenate([x, shortcut])

    x = Conv2D(filters2, (kernel_size, kernel_size),padding="same", use_bias=False)(shortcut)
    x = BatchNormalization(epsilon=eps, axis=bn_axis)(x)
    x = Scale(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters1, (1, 1), padding="same",use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis)(x)
    x = Scale(axis=bn_axis)(x)
    x = Activation('relu')(x)
 
    x = concatenate([x, input_tensor])
    return x, shortcut
    
def net(img_input, classes, blocks):
    
    x = ZeroPadding2D((1, 1))(img_input)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis)(x)
    x = Scale(axis=bn_axis)(x)
    x = advanced_activations.LeakyReLU(alpha=0.2)(x)#

    x, shortcut = block(x, 3, [64, 64], strides = (1, 1), shortcut = None)
    for i in xrange(blocks[0]):
        x, shortcut = block(x, 3, [64, 64], strides = (1, 1), shortcut = shortcut)

    x, shortcut = block(x, 3, [128, 128], shortcut = shortcut)
    for i in xrange(blocks[1]):
        x, shortcut = block(x, 3, [128, 128], strides = (1, 1), shortcut = shortcut)

    x, shortcut = block(x, 3, [256, 256], shortcut = shortcut)

    for i in xrange(blocks[2]):
        x, shortcut = block(x, 3, [256, 256], strides = (1, 1), shortcut = shortcut)

    x, shortcut = block(x, 3, [512, 512], shortcut = shortcut)
    
    for i in xrange(blocks[3]):
        x, shortcut = block(x, 3, [512, 512], strides = (1, 1), shortcut = shortcut)
    
    x = Conv2D(512, (1, 1), use_bias=False)(x)
    
    x = BatchNormalization(epsilon=eps, axis=bn_axis)(x)
    x = advanced_activations.LeakyReLU(alpha=0.2)(x)#
    x_fc = GlobalAveragePooling2D()(x)
    
    x_fc = Dense(classes, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)

    return model

def net_beta(img_input, classes, blocks):
    
    x = ZeroPadding2D((1, 1))(img_input)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    
    x = BatchNormalization(epsilon=eps, axis=bn_axis)(x)
    x = Scale(axis=bn_axis)(x)
    x = advanced_activations.LeakyReLU(alpha=0.2)(x)#

    x, shortcut = b_block(x, 3, [64, 64], strides = (1, 1), shortcut = None)
    for i in xrange(blocks[0]):
        x, shortcut = b_block(x, 3, [64, 64], strides = (1, 1), shortcut = shortcut)

    x, shortcut = b_block(x, 3, [128, 128], shortcut = shortcut)###ci
    for i in xrange(blocks[0]):
        x, shortcut = b_block(x, 3, [128, 128], strides = (1, 1), shortcut = shortcut)
    
    x, shortcut = b_block(x, 3, [256, 256], shortcut = shortcut)
    for i in xrange(blocks[0]):
        x, shortcut = b_block(x, 3, [256, 256], strides = (1, 1), shortcut = shortcut)
    
    x, shortcut = b_block(x, 3, [512, 512], shortcut = shortcut) 
    for i in xrange(blocks[0]):
        x, shortcut = b_block(x, 3, [512, 512], strides = (1, 1), shortcut = shortcut)
    # x, shortcut = block(x, 3, [512, 512], shortcut = shortcut)

    x = Conv2D(512, (1, 1), use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis)(x)
    x = advanced_activations.LeakyReLU(alpha=0.2)(x)

    x_fc = GlobalAveragePooling2D()(x)
    x_fc = Dense(classes, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)

    return model

def casnet(img_input, classes, blocks):
    x = ZeroPadding2D((1, 1))(img_input)
    x = Conv2D(16, (3, 3), use_bias=False, kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = advanced_activations.LeakyReLU(alpha=0.2)(x)#Activation('relu')(x)

    x, shortcut = block(x, 3, [16, 16], strides = (1, 1), shortcut = None)
    for i in xrange(blocks[0]):
        x, shortcut = block(x, 3, [16, 16], strides = (1, 1), shortcut = shortcut)
    x, shortcut = block(x, 3, [16, 16], shortcut = shortcut)
    
    for i in xrange(blocks[1]):
        x, shortcut = block(x, 3, [32, 32], strides = (1, 1), shortcut = shortcut)
    x, shortcut = block(x, 3, [32, 32], shortcut = shortcut)

    for i in xrange(blocks[2]):
        x, shortcut = block(x, 3, [64, 64], strides = (1, 1), shortcut = shortcut)
    

    x = Conv2D(64, (1, 1), use_bias=False, kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis)(x)   
    x = advanced_activations.LeakyReLU(alpha=0.2)(x)#Activation('relu')(x)

    x_fc = GlobalAveragePooling2D()(x)
    x_fc = Dense(classes, activation='softmax', kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(x_fc)

    model = Model(img_input, x_fc)

    return model


"""
    NETWORK ARCHITECTURES FOR HISTOPHENOTYPES DATA. 
"""
def net18(img_input, classes = None):
    return net(img_input, classes, [1,1,1,1])

def net34(img_input, classes = None):
    return net(img_input, classes, [2,3,5,2])

def net50(img_input, classes = None):
    return net_beta(img_input, classes, [2,2,2,2])#3333

def net101(img_input, classes = None):
    return net_beta(img_input, classes, [4,5,6,5])#6676


"""
    NETWORK ARCHITECTURES FOR CIFAR-10.  
"""
def net20(img_input, classes = None):
    return casnet(img_input, classes, [3,2,1])

def net32(img_input, classes = None):
    return casnet(img_input, classes, [5,4,3])

def net44(img_input, classes = None):
    return casnet(img_input, classes, [7,6,5])

def net56(img_input, classes = None):
    return casnet(img_input, classes, [9,8,7])

def net110(img_input, classes = None):
    return casnet(img_input, classes, [18,17,16])
