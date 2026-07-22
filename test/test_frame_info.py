import sys
import clr
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

    print("===== Frame Information =====")
    print("Width:", frame.Width)
    print("Height:", frame.Height)
    print("BytesPerPixel:", frame.BytesPerPixel)
    print("PixelDataLength:", frame.PixelDataLength)

    frame.Dispose()

    sensor.Stop()

sensor.DepthFrameReady += depth_ready

sensor.Start()

while sensor.IsRunning:
    time.sleep(0.1)