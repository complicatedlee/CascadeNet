# -*- coding: utf-8 -*-

# In[1]:
import keras as K
import keras.layers as L
import tensorflow as tf
import scipy.io as sio
import argparse,os
import numpy as np
import h5py
import time
import sys
import cv2
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import classification_report 
from sklearn.metrics import cohen_kappa_score 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,TensorBoard,LearningRateScheduler
from keras.utils import plot_model, np_utils


import models as models
from data_util import MAUC, eval, color_preprocessing
from load_data import load_data, load_cifar10_data
import time

model_names = sorted(name for name in models.__dict__
if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))


parser=argparse.ArgumentParser()
parser.add_argument('--mode',
                    type=int ,
                    default=0,
                    help='0 represent train mode, 1 represent test mode.')
parser.add_argument('--epoch',
                    type=int ,
                    default=100,
                    help='train epochs')
parser.add_argument('--batch_size',
                    type=int ,
                    default=64,
                    help='batchsize')
parser.add_argument('--lr',
                    type=float,
                    default=0.1,
                    help='learning rate')
parser.add_argument('--model_name', 
                    type=str,
                    default='resnet20',  
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--dataset',
                    type=str,
                    default='cifar',
                    help='choose dataset')

args=parser.parse_args()

def model(img_rows, img_cols, channel, num_classes):
    x = L.Input(shape = (img_rows, img_cols, channel))

    model = models.__dict__[args.model_name]( x, classes = num_classes)

    opt = K.optimizers.SGD(lr=args.lr, decay=1e-4, momentum=0.9, nesterov=True)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def scheduler(epoch):
    if epoch < 60:
        return 0.01
    # elif epoch < 250:
    #     return 0.01
    return 0.001

def train(x_train, y_train , x_test, y_test, model = None):

    model_ckt = ModelCheckpoint(monitor = 'val_acc', filepath='./weights/' + args.model_name, verbose=1, save_best_only=True)
    tensorbd=TensorBoard(log_dir='./log',histogram_freq=0, write_graph=True, write_images=True)
    sch = LearningRateScheduler(scheduler)

    y_train = np_utils.to_categorical(y_train[:], num_classes)
    y_test = np_utils.to_categorical(y_test[:], num_classes)

    print('Train data shape:{}'.format(x_train.shape))

    
    # Train
    generator = ImageDataGenerator(horizontal_flip=True,
                                    width_shift_range=0.125,
                                    height_shift_range=0.125,
                                    fill_mode='constant',cval=0.
                                )
    
    generator.fit(x_train, seed=0)
    # # model.load_weights('./weights/' + args.model_name)

    # Start Fine-tuning
    model.fit_generator(generator.flow(x_train, y_train,
            batch_size=args.batch_size),
            steps_per_epoch=len(x_train)//args.batch_size + 1, 
            epochs=args.epoch,
            validation_data=(x_test, y_test),
            callbacks=[model_ckt, tensorbd, sch] 
            # K.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)]
            )

    
def test(x_test, y_test, model = model):

    print('Test data shape:{}'.format(x_test.shape))

    model.load_weights('../weights/' + args.model_name)

    x_crop_test = center_crop_image(x_test, 27)
    # Make predictions
    predictions_valid = model.predict([x_crop_test, x_test], batch_size=args.batch_size, verbose=1)

    label_predictions = zip(y_test, predictions_valid)
    auc = MAUC(label_predictions, num_classes)

    print(predictions_valid.shape,y_test.shape)
    print(auc)
    # print score
    print('OA: {}%'.format(eval(predictions_valid,y_test)))

    # generate confusion_matrix
    print("Confusion_matrix")
    prediction=np.asarray(predictions_valid)
    pred=np.argmax(prediction,axis=1)
    pred=np.asarray(pred,dtype=np.int8)
    
    print(confusion_matrix(y_test,pred))
    print(sum(np.diag(confusion_matrix(y_test,pred))))

    # generate accuracy
    print(classification_report(y_test, pred))
    print("Kappa: {}".format(cohen_kappa_score(y_test,pred)))



if __name__ == '__main__':
    
    # Load data. Please implement your own load_data() module for your own dataset
    if args.dataset == 'cell':
        print("======> Loading CELL data...")
        x_train, y_train , x_test, y_test = load_data()
    elif args.dataset == 'cifar':
        print("======> Loading CIFAR data...")
        x_train, y_train , x_test, y_test = load_cifar10_data()
        
    # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)

    #classes, channel
    num_classes = np.max(y_train) + 1
    (num, img_rows, img_cols, channel) = x_train.shape
    print(num_classes, x_train.shape)

    #Load our model
    print("======> Loading model '{}'".format(args.model_name))
    model = model(img_rows, img_cols, channel, num_classes)
    model.summary()

    if args.mode == 0:

        print("======> Painting model...")
        plot_model(model,to_file='./plt_models/' + args.model_name + 'model.png',show_shapes=True)
        print("Done." + "\n")

        print("======> Starting Training...")
        time0 = time.time()

        train(x_train, y_train , x_test, y_test, model = model)

        time1 = time.time()-time0
        print("Total time:", time1)
        print("Done." + "\n")


    else:

        print("======> Starting Testing...")
        test(x_test, y_test, model = model)
        print("Done." + "\n")



