import cv2
import numpy as np


class ObjectAnalyzer:
   
    def __init__(self, fx=285.63, fy=285.63, mad_factor=2.0, debug=False):
        self.fx = fx
        self.fy = fy
        self.mad_factor = mad_factor
        self.debug = debug
        self.erode_kernel = np.ones((3, 3), np.uint8)

    def filter_top_surface(self, diff, mask):
        object_diff = diff[mask > 0]

        if len(object_diff) == 0:
            return mask

        hist, bin_edges = np.histogram(object_diff, bins=30)
        peak_idx = np.argmax(hist)
        center = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0

        mad = np.median(np.abs(object_diff - center)) + 1e-6

        refined = np.zeros_like(mask)
        keep = (mask > 0) & (
            np.abs(diff.astype(np.float32) - center) <= self.mad_factor * 1.4826 * mad
        )
        refined[keep] = 255
        return refined

    def find_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def largest_contour(self, contours):
        if len(contours) == 0:
            return None
        return max(contours, key=cv2.contourArea)

    def oriented_dimensions(self, mask, reference_depth):
        
        clean_mask = cv2.erode(mask, self.erode_kernel, iterations=1)
        contours = self.find_contours(clean_mask)

        if len(contours) == 0:
            return 0.0, 0.0, 0.0

        largest = self.largest_contour(contours)
        (cx, cy), (w_px, h_px), angle = cv2.minAreaRect(largest)

        ys, xs = np.where(mask > 0)
        z_values = reference_depth[ys, xs].astype(np.float32)
        z_values = z_values[z_values > 0]

        if len(z_values) == 0:
            return 0.0, 0.0, angle

        z_mean = float(np.mean(z_values))

        width_mm = w_px * z_mean / self.fx
        length_mm = h_px * z_mean / self.fy

        width_cm = width_mm / 10.0
        length_cm = length_mm / 10.0

        dim_a, dim_b = sorted([width_cm, length_cm], reverse=True)

        if self.debug:
            print(f"[Oriented BBox] pixel size: {w_px:.1f} x {h_px:.1f} px, "
                  f"angle: {angle:.1f} deg, mean depth: {z_mean:.0f} mm")

        return dim_a, dim_b, angle

    def draw_oriented_box(self, image, mask):
        contours = self.find_contours(mask)

        if len(contours) == 0:
            return image

        largest = self.largest_contour(contours)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        output = image.copy()
        cv2.drawContours(output, [box], 0, (0, 255, 0), 2)
        return output