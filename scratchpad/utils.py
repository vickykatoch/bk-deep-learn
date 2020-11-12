from matplotlib.pyplot import get_current_fig_manager
import numpy as np
import h5py
from os import path , getcwd




# Calculates derivative/gradient of sigmoid
# Formula : s`(z) = s(z) * (1 - s(z))
# Where : sigmoid prime of z = sigmoid of z (1 - sigmoid of z)
def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1-s)

# Converts 3 dimensional image array to single dimention vector
def image2vec(image):
    return image.reshape(image.size,1)

# Calculate row-wise norm of a matrix
# Where ||array|| = square root of(row-wise sum of squares)
def normalized_row(array):
#     norm = np.sqrt(np.sum(array * array,axis=1, keepdims=True))
    norm = np.linalg.norm(array,axis=1, keepdims=True)
    return array/norm

def load_datasets(notebook=True):
    base_path = getcwd() if notebook==True else path.dirname(__file__)
    train_file_path = path.join(base_path,'datasets/train_catvnoncat.h5')
    # train_file_path = path.join(getcwd(),'datasets/train_catvnoncat.h5')
    train_dataset = h5py.File(train_file_path, 'r')
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_file_path = path.join(base_path,'datasets/test_catvnoncat.h5')
    test_dataset = h5py.File(test_file_path,'r')
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = np.reshape(train_set_y_orig,(1,train_set_y_orig.size))
    test_set_y_orig = np.reshape(test_set_y_orig,(1,test_set_y_orig.size))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def print_stats(train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig):
    print('======================TRAIN DATA [STARTS]===========================')
    tdim = train_set_x_orig.shape[1]
    print(f'X Samples Count\t: {train_set_x_orig.shape[0]}')
    print(f'X Image Dimention\t: {tdim} * {tdim}')
    print(f'X Shape \t: {train_set_x_orig.shape}')

    print(f'Y Samples Count\t: {train_set_y_orig.shape[1]}')
    print(f'Y Shape \t: {train_set_y_orig.shape}')

    print('======================TRAIN DATA [STARTS]===========================\n')

    print('======================TEST DATA [STARTS]===========================')
    tdim = test_set_x_orig.shape[1]
    print(f'X Samples Count\t: {test_set_x_orig.shape[0]}')
    print(f'X Image Dimention\t: {tdim} * {tdim}')
    print(f'X Shape \t: {test_set_x_orig.shape}')

    print(f'Y Samples Count\t: {test_set_y_orig.shape[1]}')
    print(f'Y Shape \t: {test_set_y_orig.shape}')
    print('======================TEST DATA [STARTS]===========================\n')







