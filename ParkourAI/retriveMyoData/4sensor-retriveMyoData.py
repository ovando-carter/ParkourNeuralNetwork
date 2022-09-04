##################################################################################################################
# Creator: Ovando Carter
# Retrives Myo Sensor data from Influxdb
# Creates a csv file from data, and saves it to the same file.
##################################################################################################################


# Initialise the client
# Is this the correct way to initialise if I want to read data? 
from datetime import datetime

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# show data retrived for testing phase 
import matplotlib.pyplot as plt
import csv

# You can generate a Token from the "Tokens Tab" in the UI
token = "_f2kBWOx9NVO82gCsNVy5O77wyd68IsbDnmKka1784_9n93rYSBq9xktWDU-vYfcrOX0-za7Zy7s12Jbt4Us2w=="
org = "londonparkourproject"
bucket = "mymacbookpro"

client = InfluxDBClient(url="http://localhost:8086", token=token)



##################################################################################################################
# example from https://docs.influxdata.com/influxdb/cloud/api-guide/client-libraries/python/

# Initiate the query client
query_api = client.query_api()

# Create Flux query to retrive the 
query1 = ' from(bucket:"mymacbookpro")\
|> range(start: -55m, stop: -47m)\
|> filter(fn:(r) => r._field == "voltage 1")'
result1 = query_api.query(org=org, query=query1)

query2 = ' from(bucket:"mymacbookpro")\
|> range(start: -55m, stop: -47m)\
|> filter(fn:(r) => r._field == "voltage 2")'
result2 = query_api.query(org=org, query=query2)

query3 = ' from(bucket:"mymacbookpro")\
|> range(start: -55m, stop: -47m)\
|> filter(fn:(r) => r._field == "voltage 3")'
result3 = query_api.query(org=org, query=query3)

query4 = ' from(bucket:"mymacbookpro")\
|> range(start: -55m, stop: -47m)\
|> filter(fn:(r) => r._field == "voltage 4")'
result4 = query_api.query(org=org, query=query4)



#################################################################################################################
# there may not be a need for this stage all together,
# it seems like result1 works just as well as results1. 
# Try testing the output by using the print() command
results1 = []
results2 = []
results3 = []
results4 = []


# select the entry in the flux query result and append it to a new table results
for table in result1:
    for record in table.records:
        results1.append((record.get_value()))

for table in result2:
    for record in table.records:
        results2.append((record.get_value()))

for table in result3:
    for record in table.records:
        results3.append((record.get_value()))

for table in result4:
    for record in table.records:
        results4.append((record.get_value()))



#################################################################################################################
# Check the results live. 
print('results1: ', results1[:10])
print('results2: ', results2[:10])
print('results3: ', results3[:10])
print('results4: ', results4[:10])



# The data sets may not have the same number of list entries for 
# the same time frame. A wire my have become disconnected
#print('len(results1): ', len(results1))
#print('len(results2): ', len(results2))

'''
# testing
results1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
results2 = [29.33, 1358.75, 698.92, 312.81, 728.25, 0.0, 0.0, 0.0, 239.49, 566.96]
results3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
results4 = [29.33, 1358.75, 698.92, 312.81, 728.25, 0.0, 0.0, 0.0, 239.49, 566.96]
'''

#################################################################################################################
# Create a new csv file only with the muscle sensor data.
# 1st to check if the data looks good - view in excel
# 2nd to use as the source data for testing and training of the neural network



# takes in two arguments, one for each list of muscle data
def write_to_csv(result1, result2, result3, result4):

    with open("4sensorMuscleData.csv", "w", newline='') as csvfile:
        
        # Create headders for the csv file
        fieldnames = ['muscle_1', 'muscle_2','muscle_3', 'muscle_4']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Insert the data for each muscle into the csv file
        writer = csv.writer(csvfile)
        for i in range(0, len(results1)): # if the list is out of range it may be because the results1 and results2 have differnt lengths. You need to choose the one that has the smallest number. 
            content = [results1[i], results2[i], results3[i], results4[i]]
            writer.writerow(content)

# take in results here and insert it into the muscle sensor data
write_to_csv(results1, results2, results3, results4)


#################################################################################################################
# Let the user know that the process has completed. 
print("Succesfull")
completeDate = datetime.now()
print('Completed on date: ', completeDate)


#################################################################################################################
# plotting the points
plt.figure()
plt.subplot(311)
plt.plot(results1)
plt.text(50, 2, 'muscel_1')
plt.ylabel('mV') # naming the y axis
#plt.xlabel('Muscle Clench Test') # naming the x axis
plt.title('MyoSensor Data') # giving a title to my graph

plt.subplot(312)
plt.plot(results2, 'r') 
plt.text(50, 2, 'muscel_2')
plt.ylabel('mV') # naming the y axis

plt.subplot(313)
plt.plot(results3, 'k')
plt.text(50, 2, 'muscel_3')
plt.ylabel('mV') # naming the y axis

'''
plt.subplot(314)
plt.plot(results4, '-')
plt.text(50, 2, 'muscel_4')
plt.ylabel('mV') # naming the y axis
'''

# add lable under all graphs
plt.xlabel('Time') # naming the x axis
  
# function to print and show the plot
plt.savefig('3sensorMyo.png')
plt.show()