import numpy as np

class VolumeCalculator:

    def __init__(self, fx=285.63, fy=285.63):
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


    def calculate_bounding_box(self, object_cloud, diff = None, mask = None):
        if object_cloud.shape[0] == 0:
            return 0.0, 0.0, 0.0

        x_min, y_min = np.percentile(object_cloud[:, :2], 2, axis=0)
        x_max, y_max = np.percentile(object_cloud[:, :2], 98, axis=0)

        z_max = np.percentile(object_cloud[:, 2], 95)

        width_cm = (x_max - x_min) / 10.0
        length_cm = (y_max - y_min) / 10.0 
        if diff is not None and mask is not None:
            heights = diff[(mask > 0) & (diff > 0)]
            height_cm = np.percentile(heights, 95) / 10.0 if len(heights) else 0.0
        else:
            height_cm = 0.0

        return width_cm, length_cm, height_cm

   