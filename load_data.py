import cv2
import numpy as np
import os
from keras import backend as K
import scipy.io as sio
from keras.datasets import cifar10
from scipy.misc import imsave
import matplotlib.pyplot as plt
from scipy.misc import imsave

PATH = './data/'
TRAIN_DATA = 'cross36_1.mat'
TEST_DATA = 'cross36_2.mat'


nb_train_samples = 50000 # 3000 training samples
nb_valid_samples = 10000 # 100 validation samples


def load_data():
    """Loads CELL dataset(4 classes).
       
        train data format: N W H C
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    train_data = sio.loadmat(os.path.join(PATH + TRAIN_DATA))
    test_data = sio.loadmat(os.path.join(PATH + TEST_DATA))
    x_train = np.array(train_data['train'])
    y_train = np.array(train_data['label'])
    x_test = np.array(test_data['test'])
    y_test = np.array(test_data['label'])

    
    print("======> Loading data...")
    #shuffle data
    np.random.seed(0)
    index=np.random.permutation(x_train.shape[0])
    x_train = x_train[index,:,:,:]
    y_train = y_train[:,index]


    y_train = y_train.transpose(1, 0)
    y_test = y_test.transpose(1, 0)

    return x_train, y_train , x_test, y_test



def load_cifar10_data():

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()

    return X_train, Y_train, X_valid, Y_valid


if __name__ == "__main__":
    x_train, y_train , x_test, y_test = load_data()

    # cls_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # y = y_train.tolist()

    # for n in xrange(len(cls_name)):
        
    #     cls_list = [i for i, x in enumerate(y) if x == [n]]
    #     index = cls_list[:5]
    #     for id_n, j in enumerate(index):
    #         img = x_train[j,:,:,:]
    #         imsave("{}{}.jpg".format(cls_name[n], id_n), img)
    
