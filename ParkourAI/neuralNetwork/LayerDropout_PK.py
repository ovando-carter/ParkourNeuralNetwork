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
# Layer Dropout
##################################################################################################################  

# Dropout
class Layer_Dropout:

     #Init
     def __init__(self, rate):
          # Store rate, we invert it as for example dropout
          # of 0.1 we need success rate of 0.9
          self.rate = 1 - rate

     # Forward pass
     def forward(self, inputs, training):
          # Save input values
          self.inputs = inputs

          # If not in the training mode - return values
          if not training: 
               self.output = inputs.copy()
               return

          #Generate and save scaled mask
          self.binary_mask = np.random.binomial(1, self.rate,
                                               size = inputs.shape)/self.rate
          # Apply mask to output values
          self.output = inputs * self.binary_mask

     # Backward pass
     def backward(self, dvalues):
          # Gradient on values
          self.dinputs = dvalues * self.binary_mask


# Input "layer"
class Layer_Input:

    #Forward Pass
    def forward(self, inputs, training):
        self.output = inputs