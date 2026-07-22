import sys
import threading

import clr
import System
import numpy as np

sdk_path = r"C:\Program Files\Microsoft SDKs\Kinect\v1.8\Assemblies"

if sdk_path not in sys.path:
    sys.path.append(sdk_path)

clr.AddReference("Microsoft.Kinect")

from Microsoft.Kinect import KinectSensor, DepthImageFormat


class KinectCamera:

    def __init__(self):

        sensors = KinectSensor.KinectSensors

        if sensors.Count == 0:
            raise RuntimeError("No Kinect sensor found.")

        self.sensor = sensors[0]

        if str(self.sensor.Status) != "Connected":
            raise RuntimeError(
                f"Kinect status = {self.sensor.Status}"
            )

        self.width = 320
        self.height = 240

        self._raw = None
        self._depth = None

        self._lock = threading.Lock()

        self._started = False

    def start(self):

        if self._started:
            return

        self.sensor.DepthStream.Enable(
            DepthImageFormat.Resolution320x240Fps30
        )

        self.sensor.DepthFrameReady += self._depth_ready

        self.sensor.Start()

        self._started = True

        print("Kinect Started")

    def stop(self):

        if not self._started:
            return

        self.sensor.DepthFrameReady -= self._depth_ready

        self.sensor.Stop()

        self._started = False

        print("Kinect Stopped")

    def _depth_ready(self, sender, e):

        frame = e.OpenDepthImageFrame()

        if frame is None:
            return

        try:

            if self._raw is None:

                self._raw = System.Array.CreateInstance(
                    System.Int16,
                    frame.PixelDataLength
                )

            frame.CopyPixelDataTo(self._raw)

            # تبدیل آرایه .NET به NumPy
            raw = np.array(self._raw, dtype=np.uint16)

            # استخراج عمق (13 بیت بالا)
            depth = raw >> 3

            depth = depth.reshape(
                frame.Height,
                frame.Width
            )

            with self._lock:

                self._depth = depth

        finally:

            frame.Dispose()

    def get_depth_frame(self):

        with self._lock:

            if self._depth is None:
                return None

            return self._depth.copy()

    def is_running(self):

        return self._started
    
    def get_resolution(self):
      return self.width, self.height


    def has_frame(self):
        with self._lock:
            return self._depth is not None