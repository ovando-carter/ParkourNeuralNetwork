##################################################################################################################
# used to monitor data output

from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
##################################################################################################################
import nnfs
import os
import pickle # allows us to save the parameters of the model into a file.
import copy
#from nnfs.datasets import spiral_data, vertical_data, sine_data #imported data set from nnfs.datasets
import numpy as np



nnfs.init()

##################################################################################################################
# Loading Data ###################################################################################################

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):

    # Scan all the directories and create a list of lables
    print('\nScanning all directories and creating a list of lables \n')
    labels = os.listdir(os.path.join(path, dataset))

    #labels.sort()
    #print('labels[1:] ', labels[1:])

    # Create lists for samples and lables
    X = []
    y = []

    # For each lable folder
    print('\nReading the csv_data \n')
    # I had to use labels[1:] because there is an invisible file 
    # .DS_Store in the folder that kept creating NotADirectoryError: [Errno 20]
    for label in labels[1:]:
        # And for each csv_data in given folder
        # again I needed to remove .DS_Store by using [1:]
        #if label in lables[1:] != '.DS_Store': # will need to delete this file
        print('label: ', label)
        for file in os.listdir(os.path.join(
            path, dataset, label
            ))[1:]:

            # Read the csv_data
            csv_data = np.loadtxt(os.path.join(path, dataset, label, file), delimiter=",")

            print('\ncsv_data: ', csv_data)

            # And append it and a label to the lists
            X.append(csv_data)
            y.append(label)
    
    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y)#.astype('uint8')

# MNIST dataset (train + test)
def create_data_mnist(path):

    # Load both sets separately
    print('Loading traning data \n')
    X, y = load_mnist_dataset('train', path)
    print('loading test data \n')
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test 