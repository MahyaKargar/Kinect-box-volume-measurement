import sys
import clr

sdk_path = r"C:\Program Files\Microsoft SDKs\Kinect\v1.8\Assemblies"

if sdk_path not in sys.path:
    sys.path.append(sdk_path)

clr.AddReference("Microsoft.Kinect")

from Microsoft.Kinect import (
    KinectSensor,
    DepthImageFormat
)

sensor = KinectSensor.KinectSensors[0]

print("Sensor Status:", sensor.Status)

sensor.DepthStream.Enable(
    DepthImageFormat.Resolution320x240Fps30
)

print("Depth Stream Enabled!")

sensor.Start()

print("Sensor Started!")

input("Press Enter to stop...")

sensor.Stop()

print("Sensor Stopped")