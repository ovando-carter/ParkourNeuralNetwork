##################################################################################################################
# Creator: Ovando Carter - using "Neural Networks From Scratch by Harrison Kingsley"
# Including:
# Modeling neurons and building layers *
# using dot product of a layer of neurons *
# Use of generated data *
# Use of test data *
# including activation function - ReLU *
#                               - Softmax
# Use of Catagorical Crossentropy Loss functions
# Optimisation - adjust the weights and biases to decrease the loss. *
# Back Propigation using chain rule *
# Optimisers - Stochastic Gradient Descent (SGD) -> using momentums
#            - Adaptive gradient​ (AdaGrad)
#            - Root Mean Square Propagation​ (​RMSProp​)
#            - Adaptive Momentum (Adam) *
# Dropout 
# Binary Crossentropy *
# The two main methods for calculating error in regression are ​
#             - mean squared error​ (MSE) 
#             - and ​mean absolute error​ (MAE).
# Imports and shuffesl data from directories using:
#             - test data
#             - training data
# Model Evaluation
# Predictions

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
import cv2 # used for solving computer vission problems 
import numpy as np



nnfs.init()

# get the functions for weights
from Initialise_weights_PK import layer_Dense

from LayerDropout_PK import Layer_Dropout, Layer_Input
from ActivationFn_PK import Activation_ReLU, Activation_Softmax, Activation_Sigmoid, Activation_Linear
from Optimizers_PK import Optimizer_SGD, Optimizer_Adagrad, Optimizer_RMSprop, Optimizer_Adam
from loss_functions_PK import Loss, Loss_CategoricalCrossentropy, Activation_Softmax_Loss_CategoricalCrossentropy, loss_BinaryCrossentropy, Loss_MeanSquaredError, Loss_MeanAbsoluteError
from Common_accuracy_class_PK import Accuracy, Accuracy_Categorical, Accuracy_Regression
from Model_PK import Model
from Loading_Data_PK import load_mnist_dataset, create_data_mnist



##################################################################################################################  
# Input data
##################################################################################################################  

# we can load our data by doing
X, y, X_test, y_test = create_data_mnist('parkour_data')

##################################################################################################################

# Shuffle the training dataset
print('Shuffeling the training dataset')
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

##################################################################################################################
#removed scaling code
'''
# Scale features

# scale images to be between the range of -1 and 1 by taking each pixel value, 
# subtracting half the maximum of all pixel values (i.e., 255/2 = 127.5). 
# NB: We could also scale our data between 0 and 1 by simply dividing it by 255 (the maximum value).
print('Scaling both training and test data')
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5)/127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5)/127.5
'''


##################################################################################################################
#
#
#
#
#
#
#
# 
##################################################################################################################  
# Artifical Neural Layers
# Model Object
# 1x512 densely-connected neural network (2 hidden layers with 512 neurons) 
##################################################################################################################  

#print("\n X.shape:", X.shape) 
#print("\n X.shape[1]:", X.shape[1]) # gives the shape as (3,), which is strange since we have 737 rows and 4 columns
#print()



# Instantiate the model
model = Model()

# add layers
model.add(layer_Dense(X.shape[1], 128)) # input layer
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1)) # dropout layer
model.add(layer_Dense(128,128)) # 1st hidden layer
model.add(Activation_ReLU()) 
model.add(Layer_Dropout(0.1)) # dropout layer
model.add(layer_Dense(128,10)) # 2nd hidden layer
model.add(Activation_Softmax()) # output layer

#print(model.layers)

# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_CategoricalCrossentropy(), #loss=Loss_MeanSquaredError(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical() #accuracy=Accuracy_Regression()
)

# Finalize the model
model.finalize()


# Train the model
model.train(X, y, validation_data = (X_test, y_test), 
            epochs=10, batch_size=128, print_every=100)


# Retrive and print parameters
parameters = model.get_parameters()


# New model

# Instance the model
model = Model()

# add layers
model.add(layer_Dense(X.shape[1], 128)) # input layer
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1)) # dropout layer
model.add(layer_Dense(128,128)) # 1st hidden layer
model.add(Activation_ReLU()) 
model.add(Layer_Dropout(0.1)) # dropout layer
model.add(layer_Dense(128,10)) # 2nd hidden layer
model.add(Activation_Softmax()) # output layer



# Set loss and accuracy objects
model.set(
    loss=Loss_CategoricalCrossentropy(), #loss=Loss_MeanSquaredError(),
    accuracy=Accuracy_Categorical() #accuracy=Accuracy_Regression()
)

# Finalize the model
model.finalize()



# Set model with parameters instead of training it
model.set_parameters(parameters)

##################################################################################################################  
# Saving and loading the parameters

# Save paraters (weights and biasies)
#model.save_parameters('parkour_data.parms')

# Load saved parameters (weights and biasies)
#model.load_parameters('parkour_data.parms')

##################################################################################################################  
# Saving and loading the model

# Save model
#model.save('parkour_data.model')

#############################################################################################################################################
# Present the name of the prediction

# I will need to change this according to the parkour move that I want the system to recognise
parkour_data_labels = {
    0: 'Kong-Professional',
    1: 'Kong-Elite',
    2: 'Kong-Experianced',
    3: 'Kong-Intermediate', 
    4: 'Kong-Beginner'
}


##################################################################################################################  
# Get image for prediction

# get image data and change it to grey scale
user_data = np.loadtxt( 'prediction_data/4MyoData.csv', delimiter=",")

#removed scaling code
'''
# Resize the plot so that it is the same size as the test data images
image_data = cv2.resize(image_data, (28, 28))

# invert the pixels so that they look like the images in the test data i.e. black background with white clothing.
image_data = 255 - image_data

# Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32)) - 127.5 /127.5
'''

##################################################################################################################  
# Load model
model = Model.load('parkour_data.model')

# Evaluate the model
#model.evaluate(X_test, y_test) 

##################################################################################################################  
# Confidences 

# Predict on the first 5 samples from validation dataset
#confidences = model.predict(X_test[:5])

# Predict on the image
confidences = model.predict(user_data)

# Print the confidence result
#print('confidences: ', confidences)



##################################################################################################################  
# Predictions

predictions = model.output_layer_activation.predictions(confidences)
#print('predictions: ', predictions) # will show only the numbers so we need to convert the class number back to the name of what it is



# Get label name from label index
prediction = parkour_data_labels[predictions[0]]

print(prediction)


