import sys
import clr
import time

sdk_path = r"C:\Program Files\Microsoft SDKs\Kinect\v1.8\Assemblies"

if sdk_path not in sys.path:
    sys.path.append(sdk_path)

clr.AddReference("Microsoft.Kinect")

from Microsoft.Kinect import (
    KinectSensor,
    DepthImageFormat
)

sensor = KinectSensor.KinectSensors[0]

sensor.DepthStream.Enable(
    DepthImageFormat.Resolution320x240Fps30
)

frame_count = 0

def depth_ready(sender, e):
    global frame_count

    frame = e.OpenDepthImageFrame()

    if frame is None:
        return

    frame_count += 1
    print("Depth Frame:", frame_count)

    frame.Dispose()

sensor.DepthFrameReady += depth_ready

sensor.Start()

print("Waiting for depth frames...")

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass

sensor.Stop()