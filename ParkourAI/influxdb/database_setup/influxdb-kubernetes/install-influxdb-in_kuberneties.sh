#! /bin/bash

kubectl create secret generic influxdb-creds \
  --from-literal=INFLUXDB_DB=monitoring \
  --from-literal=INFLUXDB_USER=user \
  --from-literal=INFLUXDB_USER_PASSWORD=<password> \
  --from-literal=INFLUXDB_READ_USER=readonly \
  --from-literal=INFLUXDB_USER_PASSWORD=<password> \
  --from-literal=INFLUXDB_ADMIN_USER=root \
  --from-literal=INFLUXDB_ADMIN_USER_PASSWORD=<password> \
  --from-literal=INFLUXDB_HOST=influxdb  \
  --from-literal=INFLUXDB_HTTP_AUTH_ENABLED=true