#Use YAML to write a docker-compose.yml file.
# within the YAML file write the following: 

version: '3.3'

services: 
  influxdb:
    image: influxdb:latest
    ports: 
      - '8086:8086'
    volumes: 
      - influxdb-storage:/var/lib/influxdb
    environment: 
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=changemeplease
      - DOCKER_INFLUXDB_INIT_ORG=londonparkourproject
      - DOCKER_INFLUXDB_INIT_BUCKET=mymacbookpro

volumes:
  influxdb-storage:


# Navigate to the directory using terminal and then type: 
docker-compose up

# This should run the database from the port 8080. Check by typing localhost:8080 into the chrome search bar. 

#Ensure you have telegraf by using brew to install
brew install telegraf

brew services start telegraf

While on the localhost:8080 you should see influxdb. You can navigate through the database in the following way: 

Data
	>Telegraf
		>create configuration
			> system > continue
				> name configuration
					> create and verify
						> finish
		> Navigate to new system
			> Set up instructions
				> Generate New token
					> copy the first API token code and paste in terminal
					> copy second token and paste in terminal
Explore
		> Search for a bucket
				>filter 
					> click mem
						> Click host
							> click used_percent
								> submit
Boards 
	> click on My mackbook usage (see the data in the database here)
		

# Now you need to set up a python file that has the ability to send data to the database. 
# Create a python file called send_data_to_influxdb.py.
# Within the influxdb database, go to Data.


Data
	> Python
		> Look at Install backage and run: pip install influxdb-client
		> look at "Initialize the Client" > copy to clipboard > paste in python	
		> look at "Write Data" > copy to clipboard > paste in python	
		> save python code
		
# Your python code should look like this: 

from datetime import datetime

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# You can generate a Token from the "Tokens Tab" in the UI
token = "_f2kBWOx9NVO82gCsNVy5O77wyd68IsbDnmKka1784_9n93rYSBq9xktWDU-vYfcrOX0-za7Zy7s12Jbt4Us2w=="
org = "londonparkourproject"
bucket = "mymacbookpro"

client = InfluxDBClient(url="http://localhost:8086", token=token)


write_api = client.write_api(write_options=SYNCHRONOUS)

data = "mem,host=host1 used_percent=23.43234543"
write_api.write(bucket, org, data)	

# Notice that the data says, 
#	> used_percent=23.43234543
#	> and host=host1
# So you will be looking for this information when you are in the influxdb page.

# Run the python code from terminal
python3 send_data_to_influxdb.py	
	
# Check on influxdb if the data was stored in the database. 
	
Explore
		> Search for a bucket
				>filter 
					> click mem
						> Click host
							> click used_percent
								> submit
								
# If sucessful, the data should show that host1 will have a used_percent of 23.43234543.