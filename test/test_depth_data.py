import sys
import clr
import System
import time

sdk_path = r"C:\Program Files\Microsoft SDKs\Kinect\v1.8\Assemblies"

if sdk_path not in sys.path:
    sys.path.append(sdk_path)

clr.AddReference("Microsoft.Kinect")

from Microsoft.Kinect import KinectSensor, DepthImageFormat

sensor = KinectSensor.KinectSensors[0]

sensor.DepthStream.Enable(
    DepthImageFormat.Resolution320x240Fps30
)

printed = False

def depth_ready(sender, e):
    global printed

    if printed:
        return

    frame = e.OpenDepthImageFrame()

    if frame is None:
        return

    printed = True

    raw = System.Array.CreateInstance(
        System.Int16,
        frame.PixelDataLength
    )

    frame.CopyPixelDataTo(raw)

    print("Type:", type(raw))
    print("Length:", len(raw))
    print("First Value:", raw[0])
    print("Middle Value:", raw[len(raw)//2])
    print("Last Value:", raw[-1])

    frame.Dispose()
    sensor.Stop()

sensor.DepthFrameReady += depth_ready

sensor.Start()

while sensor.IsRunning:
    time.sleep(0.1)