import numpy as np
import open3d as o3d

class Measurement:

    def __init__(self):
        self.fx = 285.63
        self.fy = 285.63

        self.cx = 160.0
        self.cy = 120.0

    def process(self, reference_depth, current_depth, mask):
        
        reference_cloud = self._depth_to_points(reference_depth)

        current_cloud = self._depth_to_points(current_depth)

        object_cloud = self._extract_object_points(current_depth, mask)

        return reference_cloud, current_cloud, object_cloud

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

        print("Mask Pixels      :", np.count_nonzero(mask))
        print("Mask > 0 Pixels  :", np.count_nonzero(mask > 0))
        print("Selected Points  :", np.count_nonzero(valid))

        return points[valid]
        

    def visualize_point_cloud(self, points, window_name="Point Cloud"):
  
        if points.size == 0:
            print("Point cloud is empty.")
            return

        point_cloud = o3d.geometry.PointCloud()

        point_cloud.points = o3d.utility.Vector3dVector(points)

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=200,
            origin=[0, 0, 0]
        )

        o3d.visualization.draw_geometries(
            [point_cloud, coordinate],
            window_name=window_name
        )