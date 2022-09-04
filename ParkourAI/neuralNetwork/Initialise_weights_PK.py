import numpy as np


##################################################################################################################
# Initialise weights
##################################################################################################################

'''
Keep values small by normalising them

randn is a gausian distribution bounded around 0. We multiplied 0.10 since some of the values that came out were
greater than one. Yet we were using a gausian distribution bounded around zero.
'''

class layer_Dense:
     def __init__(self, n_inputs, n_neurons, 
                  weight_regularizer_l1 = 0, weight_regularizer_l2 = 0, 
                  bias_regularizer_l1 = 0, bias_regularizer_l2 = 0):
          self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) 
          self.biases = np.zeros((1, n_neurons))
          # Initialize weights and biases
          self.weight_regularizer_l1 = weight_regularizer_l1
          self.weight_regularizer_l2 = weight_regularizer_l2
          self.bias_regularizer_l1 = bias_regularizer_l1
          self.bias_regularizer_l2 = bias_regularizer_l2
     
     # Forward pass
     def forward(self, inputs, training):
          self.output = np.dot(inputs, self.weights) + self.biases
          self.inputs = inputs # added based on the book. supposed to help with back propigation
     
     # Backward pass - backpropigation - derivatives
     def backward(self, dvalues):
          # Gradients on parameters
          self.dweights = np.dot(self.inputs.T, dvalues)
          self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
          
          # Gradients on regularizations
          # L1 on weights
          if self.weight_regularizer_l1 > 0:
               dl1 = np.ones_like(self.weights)
               dl1[self.weights < 0] = -1
               self.dweights += self.weight_regularizer_l1 * dl1
          #l2 on weights
          if self.weight_regularizer_l2 > 0:
               self.dweights += 2 * self.weight_regularizer_l2 *\
                    self.weights
          # L1 on biases
          if self.bias_regularizer_l1 > 0:
               dl1 = np.ones_like(self.biases)
               dl1[self.biases < 0] = -1
               self.dbiases += self.bias_regularizer_l1 * dl1
          # L2 on biases
          if self.bias_regularizer_l2 > 0: 
               self.dbiases += 2 * self.bias_regularizer_l2 * \
                    self.biases
          
          # Gradient on values
          self.dinputs = np.dot(dvalues, self.weights.T)
     
     # Retrive layer parameters
     def get_parameters(self):
          return self.weights, self.biases
     
     # Set weights and biases in a layer instance
     def set_parameters(self, weights, biases):
          self.weights = weights
          self.biases = biases