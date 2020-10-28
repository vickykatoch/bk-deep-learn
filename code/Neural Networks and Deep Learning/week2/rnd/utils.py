import numpy as np
import h5py
from os import path , getcwd
    
    
def load_dataset():
    train_file_path = path.join(getcwd(),'..','data/train_catvnoncat.h5')
    train_dataset = h5py.File(train_file_path, "r")
    # return train_dataset
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    test_file_path = path.join(getcwd(),'..','data/test_catvnoncat.h5')
    
    test_dataset = h5py.File(test_file_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes