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

# You can generate a Token from the "Tokens Tab" in the UI
token = "_f2kBWOx9NVO82gCsNVy5O77wyd68IsbDnmKka1784_9n93rYSBq9xktWDU-vYfcrOX0-za7Zy7s12Jbt4Us2w=="
org = "londonparkourproject"
bucket = "mymacbookpro"

client = InfluxDBClient(url="http://localhost:8086", token=token)



##################################################################################################################
# example from https://docs.influxdata.com/influxdb/cloud/api-guide/client-libraries/python/

# Initiate the query client
query_api = client.query_api()

# Create Flux query
query = ' from(bucket:"mymacbookpro")\
|> range(start: -1m)\
|> filter(fn:(r) => r._measurement == "muscles_sensor")'
result = query_api.query(org=org, query=query)
results = []
for table in result:
    for record in table.records:
        results.append((record.get_value()))

#print(results)


#################################################################################################################
# Create a new csv file only with the list of emails.
# Just to check if the data looks good - view in excel
def write_to_csv(muscleSensorData):
    with open("TestMuscleData.csv", 'w') as csvfile:
        csvfile.write('Volatge' + '\n') # adds a header called Volatge to the csv file
        for data in muscleSensorData:
            csvfile.write(str(data) + '\n') # takes each data point in the list and writes it on a seperate line. 

write_to_csv(results)

print("Succesfull")