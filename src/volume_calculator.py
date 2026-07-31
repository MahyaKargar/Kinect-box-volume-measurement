import numpy as np

class VolumeCalculator:

    def __init__(self, fx=585.0, fy=585.0):
        self.fx = fx
        self.fy = fy

    def calculate_volume_from_diff(self, diff, reference_depth, mask):
        valid_mask = (mask > 0) & (diff > 0)

        if not np.any(valid_mask):
            return 0.0

        heights = diff[valid_mask].astype(np.float32)
        z_ref = reference_depth[valid_mask].astype(np.float32)

        pixel_area = (z_ref / self.fx) * (z_ref / self.fy)

        total_volume_cm3 = np.sum(heights * pixel_area) / 1000.0
        return total_volume_cm3


    def calculate_bounding_box(self, object_cloud):
        if object_cloud.shape[0] == 0:
            return 0.0, 0.0, 0.0

        x_min, y_min, z_min = np.min(object_cloud, axis=0)
        x_max, y_max, z_max = np.max(object_cloud, axis=0)

        width_cm = (x_max - x_min) / 10.0
        length_cm = (y_max - y_min) / 10.0 
        height_cm = z_max / 10.0

        return width_cm, length_cm, height_cm

   