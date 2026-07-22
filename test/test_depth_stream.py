import sys
import clr

# مسیر اسمبلی SDK
sdk_path = r"C:\Program Files\Microsoft SDKs\Kinect\v1.8\Assemblies"

if sdk_path not in sys.path:
    sys.path.append(sdk_path)

clr.AddReference("Microsoft.Kinect")

from Microsoft.Kinect import KinectSensor

print("Loading Kinect...")

sensor = KinectSensor.KinectSensors[0]

print("Status:", sensor.Status)

print("Depth stream object:")
print(sensor.DepthStream)

print("Depth format:")
print(sensor.DepthStream.FramePixelDataLength)

print("Max reliable distance:")
print(sensor.DepthStream.MaxDepth)

print("Min reliable distance:")
print(sensor.DepthStream.MinDepth)