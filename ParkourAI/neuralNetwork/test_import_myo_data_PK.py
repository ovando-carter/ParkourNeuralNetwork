import numpy as np
import os

path = 'parkour_data'
dataset = 'train'


# Scan all the directories and create a list of lables
labels = os.listdir(os.path.join(path, dataset))
print(labels[1:])

# For each lable folder
print('Reading the csv_data \n')
# I had to use labels[1:] because there is an invisible file 
# .DS_Store in the folder that kept creating NotADirectoryError: [Errno 20]
for label in labels[1:]:

    print(label)

    # And for each csv_data in given folder
    for file in os.listdir(os.path.join(
        path, dataset, label
        ))[1:]:
        
        print(os.listdir(os.path.join(path, dataset, label))[1:])

        #file = os.path.join('parkour_data', 'test/0/4MyoData.csv')
        #file_and_path = '/Users/apple/Documents/coding/myProjects/Arduino_Myo_Project/ovando_development/Ovando_C++_Myo_sensor/ParkourAI/neuralNetwork/parkour_data/test/0/4MyoData.csv'
        print('file: ', os.path.join(path, dataset, label, file))
        array = np.loadtxt( os.path.join(path, dataset, label, file), delimiter=",")

        print (array)