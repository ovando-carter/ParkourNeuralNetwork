void setup() 
{
  Serial.begin(9600);
}

// Reads in the sensor data from the myoware myo-sensor
void loop() 
{
  float sensorValue = analogRead(A1);
  float millivolt = (sensorValue/1023)*5;
  
  Serial.print("Sensor Value: ");
  Serial.println(sensorValue);
  
  Serial.print("Voltage: ");
  Serial.print(millivolt*1000);
  Serial.println(" mV");
  Serial.println("");
  delay(1);       
}
