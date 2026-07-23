import numpy as np

class Measurement:

    def __init__(self):
        self.fx = 285.63
        self.fy = 285.63

        self.cx = 160.0
        self.cy = 120.0

    def process(self, refrence_depth, current_depth, mask):
        
        refrence_cloud = self._depth_to_points(refrence_depth)

        current_cloud = self._depth_to_points(current_depth)

        object_cloud = self._extract_object_points(current_depth, mask)

        return (refrence_cloud, current_cloud, object_cloud)

    def _create_points(self, depth):
        height, width = depth.shape

        u, v = np.meshgrid(
            np.arange(width),
            np.arange(height)
        )

        z = depth.astype(np.float32)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        points = np.stack(
            (
                x,
                y,
                z
            ),
            axis=-1
        )

        return points

    def _depth_to_points(self, depth):
        
        points = self._create_points(depth)
        valid = depth > 0

        return points[valid]

    def _extract_object_points(self, depth, mask):
        points = self._create_points(depth)
        valid = (depth > 0) & (mask > 0)

        return points[valid]