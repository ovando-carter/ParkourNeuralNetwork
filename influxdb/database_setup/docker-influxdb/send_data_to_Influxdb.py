from datetime import datetime

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# You can generate a Token from the "Tokens Tab" in the UI
token = "_f2kBWOx9NVO82gCsNVy5O77wyd68IsbDnmKka1784_9n93rYSBq9xktWDU-vYfcrOX0-za7Zy7s12Jbt4Us2w=="
org = "londonparkourproject"
bucket = "mymacbookpro"

client = InfluxDBClient(url="http://localhost:8086", token=token)


# Write data - sends the data to influxdb on host1
write_api = client.write_api(write_options=SYNCHRONOUS)

data = "mem,host=host1 used_percent=23.43234543"
write_api.write(bucket, org, data)