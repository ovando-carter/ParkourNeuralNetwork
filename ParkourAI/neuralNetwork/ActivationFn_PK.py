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
# Activation functions
##################################################################################################################
# Rectified Linear Activation Function
class Activation_ReLU:
     # Forward pass
     def forward(self, inputs, training):
          self.inputs = inputs # Remember input values - added to help with back propigation
          self.output = np.maximum(0, inputs)
     # Backward pass
     def backward(self, dvalues):
          # Since we need to modify the original variable, 
          # let's make a copy of the values first
          self.dinputs = dvalues.copy()
          
          # Zero gradient where input values were negative
          self.dinputs[self.inputs <= 0] = 0
     
     # Calculate predictions for outputs
     def predictions(self, outputs):
          return outputs
          

# Exponential activation function
class Activation_Softmax:
     def forward(self, inputs, training):
          # Remember input values
          self.inputs = inputs
          # Get unnormalized probabilities (axis = 1, keepdims=True used for normalisations)
          exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True)) 
          # Normalize them for each sample
          probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
          self.output = probabilities
     
         
     # Backwards pass
     def backward(self, dvalues):
          
          # Create uninitialized array
          self.dinputs = np.empty_like(dvalues)
          
          # Enumerate outputs and gradients
          for index, (single_output, single_dvalues) in \
          enumerate(zip(self.output, dvalues)):
               # Flatten output array
               single_output = single_output.reshape(-1, 1)
               #Calculate Jacobian matrix of the output and
               jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)
               #Calculate sample-wise gradient
               # and add it to the array of sample gradients
               self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
     
     # Calculate predictions for outputs
     def predictions(self, outputs):
          return np.argmax(outputs, axis=1)
         
# Sigmoid activation
class Activation_Sigmoid:

     # Forward pass
     def forward(self, inputs, training):
          # Save input and calculate/save output
          # of the sigmoid function
          self.inputs = inputs
          self.output = 1 / (1+ np.exp(-inputs))

     # Backward pass
     def backward(self, dvalues):
          # Derivative - calculates from output of the sigmoid function
          self.dinputs = dvalues * (1 - self.output) * self.output

     # Calculate predictions for outputs
     def predictions(self, outputs):
          return (outputs > 0.5) * 1

# Linear activation
class Activation_Linear:

    # Forward pass
    def forward(self, inputs, training):
        # Just remember values
        self.inputs = inputs
        self.output = inputs
    
    # backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()
    
    # Calculate predictions for outputs
    def predictions(self, outputs):
         return outputs