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
# Common accuracy class
##################################################################################################################  

class Accuracy:

     # Calculates an accuracy
     # given predictions and ground truth values
     def calculate(self, predictions, y):

          # Get comparison results
          comparisons = self.compare(predictions, y)

          # Calculate and accuracy
          accuracy = np.mean(comparisons)

          # Add accumulated sum of matching values and sample count
          self.accumulated_sum += np.sum(comparisons)
          self.accumulated_count += len(comparisons)

          # Return accuracy
          return accuracy

     # Calculate accumulated accuracy
     def calculate_accumulated(self):

          # Calculate and accuracy
          accuracy = self.accumulated_sum / self.accumulated_count

          # Return the data and regularization losses
          return accuracy
     
     # Rest variables for accumulated accuracy
     def new_pass(self):
          self.accumulated_sum = 0
          self.accumulated_count = 0


# Accuracy calcualtion for classification model
class Accuracy_Categorical(Accuracy):

    # No initialization is needed
    def init(self, y):
        pass
    
    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis = 1)
        return predictions == y


# Accuracy calculation fro regression model
class Accuracy_Regression(Accuracy):
     def __init__(self):
          #Create precision property
          self.precision = None
     
     # Calculates precision value
     # based on passed in groun truth
     def init(self, y, reinit=False):
          if self.precision is None or reinit:
               self.precision = np.std(y) / 250
     
     # Compares predictions to the ground truth values
     def compare(self, predictions, y):
          return np.absolute(predictions - y) < self.precision



