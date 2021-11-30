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
##################################################################################################################

import numpy as np
import nnfs
from nnfs.datasets import spiral_data #imported data set from nnfs.datasets
from nnfs.datasets import vertical_data # I want to use this as the out-of-sample testing data
from nnfs.datasets import sine_data # I want to use this as the out-of-sample testing data

##################################################################################################################
# used to monitor data output
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
##################################################################################################################


nnfs.init()



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
          self.weights = 0.1*np.random.randn(n_inputs, n_neurons) 
          self.biases = np.zeros((1, n_neurons))
          # Initialize weights and biases
          self.weight_regularizer_l1 = weight_regularizer_l1
          self.weight_regularizer_l2 = weight_regularizer_l2
          self.bias_regularizer_l1 = bias_regularizer_l1
          self.bias_regularizer_l2 = bias_regularizer_l2
     
     # Forward pass
     def forward(self, inputs):
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


##################################################################################################################
# Activation functions
##################################################################################################################
# Rectified Linear Activation Function
class Activation_ReLU:
     # Forward pass
     def forward(self, inputs):
          self.inputs = inputs # Remember input values - added to help with back propigation
          self.output = np.maximum(0, inputs)
     # Backward pass
     def backward(self, dvalues):
          # Since we need to modify the original variable, 
          # let's make a copy of the values first
          self.dinputs = dvalues.copy()
          
          # Zero gradient where input values were negative
          self.dinputs[self.inputs <= 0] = 0
          

# Exponential activation function
class Activation_Softmax:
     def forward(self, inputs):
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
         
# Sigmoid activation
class Activation_Sigmoid:

     # Forward pass
     def forward(self, inputs):
          # Save input and calculate/save output
          # of the sigmoid function
          self.inputs = inputs
          self.output = 1 / (1+ np.exp(-inputs))

     # Backward pass
     def backward(self, dvalues):
          # Derivative - calculates from output of the sigmoid function
          self.dinputs = dvalues * (1 - self.output) * self.output


# Linear activation
class Activation_Linear:

    # Forward pass
    def forward(self, inputs):
        # Just remember values
        self.inputs = inputs
        self.output = inputs
    
    # backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()


##################################################################################################################
# Calculating loss
##################################################################################################################
class Loss:

     # Regularization loss calculation
     def regularization_loss(self, layer):

          # 0 by default
          regularization_loss = 0

          # L1 regularization - weights
          # calculate only when factor greater than 0
          if layer.weight_regularizer_l1 > 0:
               regularization_loss += layer.weight_regularizer_l1 *\
                    np.sum(np.abs(layer.weights))

          # L2 regularization - weights
          if layer.weight_regularizer_l2 > 0:
               regularization_loss += layer.weight_regularizer_l2 *\
                    np.sum(layer.weights *\
                         layer.weights)

          # L1 regularization - biases
          # calculate only when factor greater than 0
          if layer.bias_regularizer_l1 > 0:
               regularization_loss += layer.bias_regularizer_l1 *\
                    np.sum(np.abs(layer.biases))     

          # L2 regularization - biases
          if layer.bias_regularizer_l2 > 0:
               regularization_loss += layer.bias_regularizer_l2 *\
                    np.sum(layer.biases * \
                         layer.biases)

          return regularization_loss 

     # Calculates the data and regularization losses
     # given model output and ground truth values
     def calculate(self, output, y):
          # Calculate sample loss
          sample_losses = self.forward(output, y)

          # Claculate mean loss
          data_loss = np.mean(sample_losses)
          
          # Return loss
          return data_loss
         
class Loss_CategoricalCrossentropy(Loss):
     # Forward pass
     def forward(self, y_pred, y_true):
          # Number of samples in a batch
          samples = len(y_pred) #want to know the total length
          # Vlip data to prevent division by 0
          # Clip both sides to not drag mean towards any value
          y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
          
          # Probabilities for target values -
          # only if categorical labels
          if len(y_true.shape) == 1:#this means they have passed scalar values
           correct_confidences = y_pred_clipped[range(samples), y_true]
          
          # Mask values - only for one-hot encoded labels
          elif len(y_true.shape) == 2:# this is for vectors
           correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
          
          # Losses 
          negative_log_likelihoods = -np.log(correct_confidences)
          return negative_log_likelihoods
     # Backward pass
     def bakward(self, dvalues, y_true):
          
          # Number of samples
          samples = len(dvalues)
          # Number of labels in every sample
          # We'll use the first sample to count them
          labels = len(dvalues[0])
          
          # If lables are sparse, turn them into one-hot vector
          if len(y_true.shape) == 1:
               y_true = np.eye(labels)[y_true]
               
          # Calculate gradient
          self.dinputs = -y_true / dvalues
          # Normalize gradient
          self.dinputs = self.dinputs / samples

# Binary cross-entropy loss
class loss_BinaryCrossentropy(Loss):

     # Forward pass
     def forward(self, y_pred, y_true):

          # Clip data to prevent division by 0
          # Clip both sides to not drag mean towards any value
          y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

          # Calculate sample-wise loss
          sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
          sample_losses = np.mean(sample_losses, axis = -1)

          # Return losses
          return sample_losses
     
     # Backward pass
     def backward(self, dvalues, y_true):

          # Number of samples
          samples = len(dvalues)
          # Number of outputs in every sample
          # We'll use the first sample to count them
          outputs = len(dvalues[0])

          # Clip data to prevent division by 0
          # Clip both sides to not drag mean towards any value
          clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

          # Calculate gradient
          self.dinputs = -(y_true / clipped_dvalues - (1 - y_true)/(1 - clipped_dvalues)) / outputs

          # Normalize gradient
          self.dinputs = self.dinputs / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
     
     # Creates activation and loss function objects
     def __init__(self):
          self.activation = Activation_Softmax()
          self.loss = Loss_CategoricalCrossentropy()
          
     # Forward pass
     def forward(self, inputs, y_true):
          # Output layer's activation function
          self.activation.forward(inputs)
          # Set the output
          self.output = self.activation.output
          # Calculate and return loss value
          return self.loss.calculate(self.output, y_true)
     
     # Backward pass
     def backward(self, dvalues, y_true):
          
          # Nuber of samples
          samples = len(dvalues)
          # If lables are one-hot encoded,
          # trun them into discrete values
          if len(y_true.shape) == 2:
               y_true = np.argmax(y_true, axis = 1)
          
          # Copy so we can safely modify
          self.dinputs = dvalues.copy()
          # Calculate gradients
          self.dinputs[range(samples), y_true] -= 1
          #Normalize gradient
          self.dinputs = self.dinputs / samples





##################################################################################################################  
# calculate errors
##################################################################################################################  

#Mean Squared Error Loss
class Loss_MeanSquaredError(Loss): # L2 loss

    # Froward pass
    def forward(self, y_pred, y_true):

        # calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        # Return losses
        return sample_losses
    
    # Backward pass 
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues)/outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples




# Mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss): # L1 loss
    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis = -1)

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):

        #Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues)/outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


##################################################################################################################  
# ​Stochastic Gradient Descent (SGD)​
##################################################################################################################  


class Optimizer_SGD:
     # Initialize optimizer - set settings,
     # learning rate of 1. is default for this optimizer - changing learner rate can effect theability for the
     # model to find global minimums and not get stuck in any local minimums. The aim is to get a low loss and a higher accuracy.
     def __init__(self, learning_rate = 1.0 , decay = 0., momentum = 0.): 
          self.learning_rate = learning_rate
          self.current_learning_rate = learning_rate
          self.decay = decay
          self.iterations = 0
          self.momentum = momentum
          
     # Call once before and parameter updates - this part should reduce the learning rate with time
     def pre_update_params(self):
          if self.decay:
               self.current_learning_rate = self.learning_rate * \
                                            (1. / (1. +  self.decay * self.iterations))
     
     # Update parameters     
     def update_params(self, layer):
          # If we use momentum
          if  self.momentum:
               # If layer does not contain momentum arrays, create them
               # filled with zeros
               if not hasattr(layer, 'weight_momentums'):
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    #  If there is no momentum array for weights
                    #  The array doesn't exist for biases yet either.
                    layer.bias_momentums = np.zeros_like(layer.biases)
               
               # Build weight updates with momentum - take previous
               # updates multiplied by retain factor and update with
               # current gradients
               weight_updates = \
                                self.momentum * layer.weight_momentums - \
                                self.current_learning_rate * layer.dweights
               layer.weight_momentums = weight_updates
               
               # Build bias updates
               bias_updates = \
                              self.momentum * layer.bias_momentums - \
                              self.current_learning_rate * layer.dbiases
               layer.bias_momentums = bias_updates
               
          # Vanilla SGD updates (as before momentum update)
          else:
               weight_updates = -self.current_learning_rate * \
                                layer.dweights
               bias_updates = -self.current_learning_rate * \
                              layer.dbiases
          # Update weights and biases using either
          #vanilla or momentum updates
          #layer.weights += -self.learning_rate * layer.dweights
          #layer.biases += - self.learning_rate * layer.dbiases
          layer.weights += weight_updates
          layer.biases += bias_updates
     
     # Call once after parameter updates
     def post_update_params(self):
          self.iterations += 1
          
##################################################################################################################  
# ​Adaptive gradient​ (AdaGrad)
##################################################################################################################  
class Optimizer_Adagrad:
     # Initialize optimizer - set settings,
     # learning rate of 1. is default for this optimizer - changing learner rate can effect theability for the
     # model to find global minimums and not get stuck in any local minimums. The aim is to get a low loss and a higher accuracy.
     def __init__(self, learning_rate = 1. , decay = 0., epsilon = 1e-7): 
          self.learning_rate = learning_rate
          self.current_learning_rate = learning_rate
          self.decay = decay
          self.iterations = 0
          self.epsilon = epsilon
          
     # Call once before and parameter updates - this part should reduce the learning rate with time
     def pre_update_params(self):
          if self.decay:
               self.current_learning_rate = self.learning_rate * \
                                            (1. / (1. +  self.decay * self.iterations))
     
     # Update parameters     
     def update_params(self, layer):
          
          # If layer does not contain cache arrays,
          # create them filled with zeros
          if not hasattr (layer, 'weight_cache'):
               layer.weight_cache = np.zeros_like(layer.weights)
               layer.bias_cache = np.zeros_like(layer.biases)
          
          # Update cache with squared current gradients
          layer.weight_cache += layer.dweights**2
          layer.bias_cache += layer.dbiases**2
          
          # Vanilla SGD parameter update + normlization
          # with square roted cache
          layer.weights += -self.current_learning_rate * \
                           layer.dweights/\
                           (np.sqrt(layer.weight_cache) + self.epsilon)
          layer.biases += -self.current_learning_rate * \
                          layer.dbiases /\
                          (np.sqrt(layer.bias_cache) + self.epsilon)
          
     # Call once after parameter updates
     def post_update_params(self):
          self.iterations += 1

##################################################################################################################  
# ​Root Mean Square Propagation​ (​RMSProp​)
##################################################################################################################  
class Optimizer_RMSprop:
     # Initialize optimizer - set settings,
     # learning rate of 1. is default for this optimizer - changing learner rate can effect theability for the
     # model to find global minimums and not get stuck in any local minimums. The aim is to get a low loss and a higher accuracy.
     def __init__(self, learning_rate = 0.001 , decay = 0., epsilon = 1e-7, rho = 0.9): 
          self.learning_rate = learning_rate
          self.current_learning_rate = learning_rate
          self.decay = decay
          self.iterations = 0
          self.epsilon = epsilon
          self.rho = rho
          
     # Call once before and parameter updates - this part should reduce the learning rate with time
     def pre_update_params(self):
          if self.decay:
               self.current_learning_rate = self.learning_rate * \
                                            (1. / (1. +  self.decay * self.iterations))
     
     # Update parameters     
     def update_params(self, layer):
          
          # If layer does not contain cache arrays,
          # create them filled with zeros
          if not hasattr (layer, 'weight_cache'):
               layer.weight_cache = np.zeros_like(layer.weights)
               layer.bias_cache = np.zeros_like(layer.biases)
          
          # Update cache with squared current gradients
          layer.weight_cache = self.rho * layer.weight_cache + \
                                (1 - self.rho) * layer.dweights**2
          layer.bias_cache = self.rho * layer.bias_cache + \
                                (1 - self.rho) * layer.dbiases**2
          
          # Vanilla SGD parameter update + normlization
          # with square roted cache
          layer.weights += -self.current_learning_rate * \
                           layer.dweights/\
                           (np.sqrt(layer.weight_cache) + self.epsilon)
          layer.biases += -self.current_learning_rate * \
                          layer.dbiases /\
                          (np.sqrt(layer.bias_cache) + self.epsilon)
          
     # Call once after parameter updates
     def post_update_params(self):
          self.iterations += 1
        
##################################################################################################################  
# ​Adaptive Momentum (Adam)
##################################################################################################################  
class Optimizer_Adam:
     # Initialize optimizer - set settings,
     # learning rate of 1. is default for this optimizer - changing learner rate can effect theability for the
     # model to find global minimums and not get stuck in any local minimums. The aim is to get a low loss and a higher accuracy.
     def __init__(self, learning_rate = 0.001 , decay = 0., epsilon = 1e-7,
                  beta_1 = 0.9, beta_2 = 0.999): 
          self.learning_rate = learning_rate
          self.current_learning_rate = learning_rate
          self.decay = decay
          self.iterations = 0
          self.epsilon = epsilon
          self.beta_1 = beta_1
          self.beta_2 = beta_2
          
     # Call once before and parameter updates - this part should reduce the learning rate with time
     def pre_update_params(self):
          if self.decay:
               self.current_learning_rate = self.learning_rate * \
                                            (1. / (1. +  self.decay * self.iterations))
     
     # Update parameters     
     def update_params(self, layer):
          
          # If layer does not contain cache arrays,
          # create them filled with zeros
          if not hasattr (layer, 'weight_cache'):
               
               layer.weight_momentums = np.zeros_like(layer.weights)
               layer.weight_cache = np.zeros_like(layer.weights)
               layer.bias_momentums = np.zeros_like(layer.biases)
               layer.bias_cache = np.zeros_like(layer.biases)
          #Update momentum with current gradients
          layer.weight_momentums = self.beta_1 * \
                                   layer.weight_momentums + \
                                   (1 - self.beta_1) * layer.dweights
          layer.bias_momentums = self.beta_1 * \
                                 layer.bias_momentums + \
                                 (1 - self.beta_1)* layer.dbiases
          # Get corrected momentum
          # Self.iteration is 0 at first pass
          # and we need to start with 1 here
          weight_momentums_corrected = layer.weight_momentums /\
                                       (1 - self.beta_1 ** (self.iterations + 1))
          bias_momentums_corrected = layer.bias_momentums / \
                                     (1 - self.beta_1 ** (self.iterations + 1))
          # Update cache with squared current gradients
          layer.weight_cache = self.beta_2 * layer.weight_cache + \
                                (1 - self.beta_2) * layer.dweights**2
          layer.bias_cache = self.beta_2 * layer.bias_cache + \
                                (1 - self.beta_2) * layer.dbiases**2
          # Get correct cache
          weight_cache_corrected = layer.weight_cache /\
                                   (1 - self.beta_2 ** (self.iterations + 1))
          bias_cache_corrected = layer.bias_cache /\
                                 (1 - self.beta_2 ** (self.iterations + 1))
          
          # Vanilla SGD parameter update + normlization
          # with square roted cache
          layer.weights += -self.current_learning_rate * \
                           weight_momentums_corrected/\
                           (np.sqrt(weight_cache_corrected) + self.epsilon)
          layer.biases += -self.current_learning_rate * \
                          bias_momentums_corrected /\
                          (np.sqrt(bias_cache_corrected) + self.epsilon)
          
     # Call once after parameter updates
     def post_update_params(self):
          self.iterations += 1


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
     def forward(self, inputs):
          # Save input values
          self.inputs = inputs
          #Generate and save scaled mask
          self.binary_mask = np.random.binomial(1, self.rate,
                                               size = inputs.shape)/self.rate
          # Apply mask to output values
          self.output = inputs * self.binary_mask

     # Backward pass
     def backward(self, dvalues):
          # Gradient on values
          self.dinputs = dvalues * self.binary_mask











##################################################################################################################  
# Input data
##################################################################################################################  

# using grenerated data as the source for input data - training data. 
#X,y = spiral_data(samples = 100, classes = 2)

# create test dataset
#X, y = vertical_data(samples = 100, classes = 3)

# Create test dataset
X,y = sine_data()


##################################################################################################################  
# Reshape data
##################################################################################################################  

# Reshape lables to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y = y.reshape(-1,1)

     
##################################################################################################################  
# Artifical Neural Layers
# make a 1x64 densely-connected neural network (1 hidden layer with 64 neurons) 
##################################################################################################################

# First neuron layer
dense1 = layer_Dense(1,64, weight_regularizer_l2 = 5e-4, bias_regularizer_l2 = 5e-4)                          
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU() # our activation function for layer 1

# Create dropout layer
dropout1 = Layer_Dropout(0.1)

# takes in the output of the previous layer and give 1 output
dense2 = layer_Dense(64,64)
# Create ReLU activation (to be used with Dense layer):
activation2 = Activation_ReLU()

# Output neuron layer ###########################################################################################
# Create third Dense layer with 64 input features (as we take output
# of previous layer here) and 1 output value
dense3 = layer_Dense(64,1)
#activation3 = Activation_Softmax() # our activation function for layer 1
#activation3 = Activation_Sigmoid()
activation3 = Activation_Linear()


#################################################################################################################



# Create Softmax classifier's combined loss and activation
#loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

#Create loss function - to use instead of Softmax when we have a binary output neuron
#loss_function = loss_BinaryCrossentropy()

# Create loss function - for use with error calculation
loss_function = Loss_MeanSquaredError()


'''
NB: dense2 has an input of 3 since the last layers output was 3, has an output of 3 since we have 3 classes. 
'''
#################################################################################################################
# Create an optimiser object
# To avoid our model getting stuck in some local minimum we can try to decay a bit slower by making our decay a
# smaller number. let’s go with 1e-3 (0.001):
#optimizer = Optimizer_SGD(decay=1e-3, momentum = 0.9) # for momentum, i.e. Stochastic Gradient Descent (SGD)​
#optimizer = Optimizer_Adagrad(decay=1e-4) # for Adaptive gradient​ (AdaGrad)
#optimizer = Optimizer_RMSprop(learning_rate = 0.02, decay=1e-5, rho = 0.999) # for Root Mean Square Propagation​ (​RMSProp​)
optimizer = Optimizer_Adam(learning_rate = 0.005, decay=1e-3) # for ​Adaptive Momentum (Adam) - the example wanted to tweak for learning_rate = 0.05, decay=5e-5 but I managed to get worse results with these values. Not too bad but still lower. 

##################################################################################################################

# Accuracy precision for accuracy calculation
# There are no really accuracy factor for regression problem,
# but we can simulate/approximate it. We'll calculate it by checking
# how many values have a difference to their ground truth equivalent
# less than given precision
# We'll calculate this precision as a fraction of standard deviation
# of al the ground truth values
accuracy_precision = np.std(y) /250
   


# Train in loop
for epoch in range(10001):
     # Perform a forward pass of the training data through this layer 
     dense1.forward(X)
     activation1.forward(dense1.output) # takes output of the first layer here

     # Perform a forward pass through activation funtion
     # Takes the output of first dense layer here
     dropout1.forward(activation1.output)

     # perform a forward pass through the second layer
     # takes output of activation function of first layer as inputs
     dense2.forward(activation1.output)  

     # Perform a forward pass through activation function
     # takes the output of second dense layer here
     activation2.forward(dense2.output)

     # perform a forward pass through the third layer
     # takes output of activation function of second layer as inputs
     dense3.forward(activation2.output)  

     # Perform a forward pass through activation function
     # takes the output of third dense layer here
     activation3.forward(dense3.output)



     ##################################################################################################################
     # Implamenting loss
     ##################################################################################################################
     '''
     To train a model, we tweak the weights and biases to improve the model’s accuracy and confidence. To do this, we
     calculate how much error the model has. The loss function is the algorithm that quantifies how wrong a model is. 
     '''     
          
     # Perform a forward pass through the activation/loss function
     # takes the output of second dense layer here and returns loss
     #data_loss = loss_activation.forward(dense2.output, y) # for softmax
     #data_loss = loss_function.calculate(dense2.output, y) 
     data_loss = loss_function.calculate(activation3.output, y)


     ##################################################################################################################
     # Regularization penalty
     ##################################################################################################################


     # Calculate regularization penalty
     '''
     # for softmax
     regularization_loss = \
          loss_activation.loss.regularization_loss(dense1) + \
          loss_activation.loss.regularization_loss(dense2) 
     '''

     # For sigmoid
     regularization_loss = \
          loss_function.regularization_loss(dense1) + \
          loss_function.regularization_loss(dense2) + \
          loss_function.regularization_loss(dense3)
     
     # Calculate overall loss
     loss = data_loss + regularization_loss

     


     ##################################################################################################################
     # Claculating accuracy
     ##################################################################################################################

     '''
     # For softmax
     # Calculate accuracy from output of activation2 and targets
     # calculate values along first axis
     predictions = np.argmax(loss_activation.output, axis = 1)
     if len(y.shape) == 2:
          y = np.argmax(y, axis = 1)
     accuracy = np.mean(predictions == y)
     '''

     '''   
     # For Sigmoid
     # Calculate accuracy from output of activation2 and targets
     # Part in the brackets returns a binary mask - array consisting of
     # True/False values, multiplying it by 1 changes it into array
     # of 1s and 0s
     predictions = (activation2.output > 0.5) * 1
     accuracy = np.mean(predictions == y)
     '''

     # Calculate accuracy from output of activation2 and targets
     # To calculate it we're taking absolute difference between
     # predictions and ground truth values and compare if differences
     # are lower than given precision value
     predictions = activation3.output
     accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)   

     
     if not epoch % 100:
          
          print(f'epoch: {epoch}, ' +
                f'acc: {accuracy:.2f}, '+
                f'loss: {loss:.3f}, ' + 
                f'data_loss: {data_loss:.3f}, '+
                f'reg_loss: {regularization_loss:.3f}, ' +
                f'lr: {optimizer.current_learning_rate:.4f} '  )
          # it would be good if i could prent this output in graphical form.    
          
          #accuracy_list.append(accuracy,)
          

     '''
     # Backward pass softmax
     #loss_activation.backward(loss_activation.output, y) 
     #dense2.backward(loss_activation.dinputs)
     #dropout1.backward(dense2.dinputs)
     activation1.backward(dense2.dinputs)
     dense1.backward(activation1.dinputs)
     '''
     #Backward pass for sigmoid and MeanSquaredError
     loss_function.backward(activation3.output, y) 
     activation3.backward(loss_function.dinputs)
     dense3.backward(activation3.dinputs)
     activation2.backward(dense3.dinputs)
     dense2.backward(activation2.dinputs)
     activation1.backward(dense2.dinputs)
     dense1.backward(activation1.dinputs)
          
          
     #Print gradients - check if the weights are being automatcially generated by the code
     '''
     print("\n" + "Layer1 weights: " + str(dense1.dweights)
           + "\n" 
           + "\n" + "Layer1 biases: " + str(dense1.dbiases)
           + "\n" 
           + "\n" + "Layer2 weights: " + str(dense2.dweights)
           + "\n" 
           + "\n" + "Layer2 biases: " + str(dense2.dbiases))
     '''


     # Use our optimazer to update weights and biases:      
     # Update weights and biases     
     optimizer.pre_update_params()
     optimizer.update_params(dense1)
     optimizer.update_params(dense2)
     optimizer.update_params(dense3)
     optimizer.post_update_params()

#print(np.around(accuracy_list, 2)) # prints the accuracy_list vlaues to 2 decimal places
# it did not append each value individually, it put each value in the same position.

'''
x_vals = []
y_vals = []

index = count()

def animate(i):
    x_vals.append(next(index))
    y_vals.append(accuracy_list) # will create a random integer between 0 and 5.
    plt.cla() # Clears out previous data so that we do not get data stacked on top of each other.
    plt.plot(x_vals, y_vals)
    
ani = FuncAnimation(plt.gcf(), animate, interval = 1000) # .gfc means get current figure, and interval = 1000 means it will run every 1 seccond

plt.tight_layout
plt.show()

# use ctrl + z to escape the running of this code.
'''


'''
# For spiral data
plt.scatter(X[:,0], X[:,1], c = y, s = 40, cmap = 'brg')
plt.plot(X, activation2.output)
plt.show()
'''
'''
X_test, y_test = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)
'''
# for sine wave
plt.plot(X, y)
plt.plot(X, activation3.output)
plt.show()
