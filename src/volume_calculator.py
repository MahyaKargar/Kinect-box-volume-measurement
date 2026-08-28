import math
import numpy as np


class VolumeCalculator:

    def __init__(self, fx=285.63, fy=285.63, tilt_angle_deg=50):
        self.fx = fx
        self.fy = fy
        self.set_tilt_angle(tilt_angle_deg)

    def set_tilt_angle(self, tilt_angle_deg):
        self.tilt_angle_deg = float(tilt_angle_deg)
        self.tilt_correction_factor = math.cos(
            math.radians(self.tilt_angle_deg)
        )

    def calculate_volume_from_diff(self, diff, reference_depth, mask):
        valid_mask = (mask > 0) & (diff > 0)

        if not np.any(valid_mask):
            return 0.0

        heights = (
            diff[valid_mask].astype(np.float32) * self.tilt_correction_factor
        )
        z_ref = reference_depth[valid_mask].astype(np.float32)

        pixel_area = (z_ref / self.fx) * (z_ref / self.fy)
        total_volume_cm3 = np.sum(heights * pixel_area) / 1000.0
        return total_volume_cm3

    def calculate_bounding_box(self, object_cloud, diff=None, mask=None):
        if object_cloud.shape[0] == 0:
            return 0.0, 0.0, 0.0

        x_min, y_min = np.percentile(object_cloud[:, :2], 2, axis=0)
        x_max, y_max = np.percentile(object_cloud[:, :2], 98, axis=0)

        width_cm = (x_max - x_min) / 10.0
        length_cm = (y_max - y_min) / 10.0

        if diff is not None and mask is not None:
            heights = diff[(mask > 0) & (diff > 0)]
            raw_h_cm = (
                np.percentile(heights, 95) / 10.0 if len(heights) else 0.0
            )
            height_cm = raw_h_cm * self.tilt_correction_factor
        else:
            height_cm = 0.0

        return width_cm, length_cm, height_cm

    def calculate_volume_prism(self, width_cm, length_cm, height_cm):
        return width_cm * length_cm * height_cm

    def calculate_cross_section_area(self, mask, reference_depth):
        ys, xs = np.where(mask > 0)

        if len(ys) == 0:
            return 0.0

        z_values = reference_depth[ys, xs].astype(np.float32)
        z_values = z_values[z_values > 0]

        if len(z_values) == 0:
            return 0.0

        pixel_area_mm2 = (z_values / self.fx) * (z_values / self.fy)
        total_area_mm2 = np.sum(pixel_area_mm2)

        return float(total_area_mm2) / 100.0

    def calculate_trimmed_mean_height(
        self, diff, mask, lower_pct=10, upper_pct=90
    ):
        heights = diff[(mask > 0) & (diff > 0)].astype(np.float32)

        if len(heights) == 0:
            return 0.0

        low = np.percentile(heights, lower_pct)
        high = np.percentile(heights, upper_pct)

        trimmed = heights[(heights >= low) & (heights <= high)]

        if len(trimmed) == 0:
            trimmed = heights

        raw_height_cm = float(np.mean(trimmed)) / 10.0
        corrected_height_cm = raw_height_cm * self.tilt_correction_factor

        return corrected_height_cm

    def calculate_volume_footprint(self, mask, reference_depth, diff):
        area_cm2 = self.calculate_cross_section_area(mask, reference_depth)
        height_cm = self.calculate_trimmed_mean_height(diff, mask)

        return area_cm2 * height_cm, area_cm2, height_cm

    def calculate_centroid_height(self, mask, diff, patch_radius=5):
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return 0.0

        cy, cx = int(np.mean(ys)), int(np.mean(xs))

        y0, y1 = max(0, cy - patch_radius), cy + patch_radius + 1
        x0, x1 = max(0, cx - patch_radius), cx + patch_radius + 1

        patch_diff = diff[y0:y1, x0:x1]
        patch_mask = mask[y0:y1, x0:x1]

        valid = patch_diff[(patch_mask > 0) & (patch_diff > 0)]

        if len(valid) == 0:
            return 0.0

        raw_centroid_h_cm = float(np.median(valid)) / 10.0
        return raw_centroid_h_cm * self.tilt_correction_factor