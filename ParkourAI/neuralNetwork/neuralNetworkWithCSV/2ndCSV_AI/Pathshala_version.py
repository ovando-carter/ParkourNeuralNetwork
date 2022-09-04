import tensorflow as tf
import numpy as np 
import pandas as pd 

# Read CSV file as a dataframe with pandas
file_name = 'data.csv'
data = pd.read_csv('/Users/apple/Documents/coding/myProjects/Arduino_Myo_Project/ovando_development/Ovando_C++_Myo_sensor/ParkourAI/neuralNetwork/neuralNetworkWithCSV/2ndCSV_AI/' + file_name, header=None)


#Get statistical information
data.describe()
#print(data.describe())

# Convert datatype to float32
data = data.astype('float32')
#print(data.astype('float32'))

# print top records from dataframe
data.head()
#print(data.head())

# convert dataframe into a numpy array
data = data.to_numpy()

# Print shape of nump array
data.shape

# Get training dataset from numpy
# this is how they select the training and testing data
# the first 700 rows will be used for training
training = data[0:700]
# any rows after this will be used for testing
testing = data[700:]

print(training.shape)
print(testing.shape)

# seperate true labled from training and testing datasets
# -1 indicates to skip the last one
training_features = training[:,0:-1] 
testing_features = testing[:,0:-1]

# : means we are interested in all the rows, -1 means only 
# consider last column where we have true lables
training_labels = training[:,-1] 
testing_lables = testing[:,-1]

# Print shape of input that training_features and true lables that is training_lables
print(training_features.shape)
print(testing_lables.shape)

# Check input shape to neural network
training_features[0].shape
#print(training_features[0].shape)


# Import classes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

#Model definition
model = Sequential()
model.add(Input(shape = (8,))) # NB: we get the shape from here: training_features[0].shape
model.add(Dense(16, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid')) # only one neuron here as it is a binary class classification problems and the activation is sigmoid

#Configure model training
model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

#train model
model.fit(training_features, training_labels, epochs = 100, validation_data = (testing_features, testing_lables))

