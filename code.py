import sys
import clr

sdk_path = r"C:\Program Files\Microsoft SDKs\Kinect\v1.8\Assemblies"

if sdk_path not in sys.path:
    sys.path.append(sdk_path)

clr.AddReference("Microsoft.Kinect")

from Microsoft.Kinect import KinectSensor

print("Assembly loaded successfully!")

sensors = KinectSensor.KinectSensors

print("Number of sensors:", sensors.Count)

for i in range(sensors.Count):
    sensor = sensors[i]
    print(f"Sensor {i}")
    print("Status:", sensor.Status)
    print("Unique ID:", sensor.UniqueKinectId)