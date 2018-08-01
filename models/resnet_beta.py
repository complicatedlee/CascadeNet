import keras
import numpy as np
import argparse,os

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D,Deconv2D, concatenate, MaxPooling2D
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K
import time

bn_axis = 3
eps = 1.1e-5
weight_decay = 0.0001

__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56']
          

def residual_block(intput,out_channel,increase=False):
    if increase:
        stride = (2,2)
    else:
        stride = (1,1)

    pre_bn   = BatchNormalization()(intput)
    pre_relu = Activation('relu')(pre_bn)

    conv_1 = Conv2D(out_channel,kernel_size=(3,3),strides=stride,padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(weight_decay))(pre_relu)
    bn_1   = BatchNormalization()(conv_1)
    relu1  = Activation('relu')(bn_1)
    conv_2 = Conv2D(out_channel,kernel_size=(3,3),strides=(1,1),padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(weight_decay))(relu1)
    if increase:
        projection = Conv2D(out_channel,
                            kernel_size=(1,1),
                            strides=(2,2),
                            padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(weight_decay))(intput)
        block = add([conv_2, projection])
    else:
        block = add([intput,conv_2])
    return block

def residual_network(img_input,classes_num=10,stack_n=3):


    # build model
    # total layers = stack_n * 3 * 2 + 2
    # stack_n = 3 by default, total layers = 20
    x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)


    for _ in range(stack_n):
        x = residual_block(x,16,False)

    x = residual_block(x,32,True)
    for _ in range(1,stack_n):
        x = residual_block(x,32,False)
    
    x = residual_block(x,64,True)
    for _ in range(1,stack_n):
        x = residual_block(x,64,False)


    x = Conv2D(64,kernel_size=(1,1),strides=(1, 1),padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(weight_decay))(x)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # # input: 64 output: 10
    x = Dense(classes_num,activation='softmax',
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)

    model    = Model(img_input, x)

    return model

# total layers = stack_n * 3 * 2 + 2
def resnet20(img_input, classes):
    return residual_network(img_input, classes, 3)

def resnet32(img_input, classes):
    return residual_network(img_input, classes, 5)

def resnet44(img_input, classes):
    return residual_network(img_input, classes, 7)

def resnet56(img_input, classes):
    return residual_network(img_input, classes, 9)