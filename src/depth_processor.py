import os

import cv2
import numpy as np


class DepthProcessor:
  
    def __init__(self, fx=285.63, fy=285.63, baseline_mm=75.0,
                 delta_disparity=0.125, debug=False):

        self.reference = None
        self.current = None
        self.difference = None

        self.debug = debug

        self.min_height = 20.0

        self.fx = fx
        self.fy = fy
        self.baseline_mm = baseline_mm
        self.delta_disparity = delta_disparity

        self.threshold_safety_factor = 2.5

        self.edge_gradient_threshold = 150.0
        self.max_plausible_height_mm = 400.0
        self.max_min_height_ceiling_mm = 60.0

        self.open_kernel = np.ones((5, 5), np.uint8)
        self.close_kernel = np.ones((5, 5), np.uint8)


    def set_reference(self, depth):
        self.reference = depth.copy()
        self._calibrate_noise_threshold()

    def set_current(self, depth):
        self.current = depth.copy()

    def has_reference(self):
        return self.reference is not None

    def has_current(self):
        return self.current is not None

    def save_reference(self, filename="reference.npy"):
        if self.reference is None:
            raise RuntimeError("Reference frame is not available.")
        np.save(filename, self.reference)

    def load_reference(self, filename="reference.npy"):
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        self.reference = np.load(filename)
        self._calibrate_noise_threshold()


    def _calibrate_noise_threshold(self):
        valid_pixels = self.reference[self.reference > 0]

        if len(valid_pixels) == 0:
            return

        blur = cv2.medianBlur(self.reference, 5)
        noise_map = cv2.absdiff(self.reference, blur)

        is_edge = self._detect_scene_edges(self.reference)
        flat_region = (self.reference > 0) & (~is_edge)

        if np.count_nonzero(flat_region) > 0:
            valid_noise = noise_map[flat_region]
        else:
            valid_noise = noise_map[self.reference > 0]

        std_noise = np.std(valid_noise)
        self.min_height = min(
            self.max_min_height_ceiling_mm,
            max(20.0, 2.0 * std_noise)
        )

        edge_ratio = 1.0 - (np.count_nonzero(flat_region) / len(valid_pixels))

        if self.debug:
            print(f"[Auto-Threshold] Base noise std (edge-filtered): {std_noise:.2f} mm")
            print(f"[Auto-Threshold] Scene edge pixels excluded: {edge_ratio * 100:.1f}%")
            print(f"[Auto-Threshold] Global min_height floor set to: {self.min_height:.2f} mm")

    def expected_resolution(self, z):
        z = z.astype(np.float32)
        res = (z ** 2) / (self.fx * self.baseline_mm) * self.delta_disparity
        res[z <= 0] = 0.0
        return res

    def _detect_scene_edges(self, depth):
        depth_f = depth.astype(np.float32)

        grad_x = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        return gradient_mag > self.edge_gradient_threshold


    def create_valid_mask(self, reference, current):
        return (reference > 0) & (current > 0)

    def median_filter(self, depth):
        return cv2.medianBlur(depth, 5)

    def remove_noise(self, depth):
        depth_clean = depth.copy()
        zero_mask = (depth_clean == 0)
        depth_clean = self.median_filter(depth_clean)
        depth_clean[zero_mask] = 0
        return depth_clean

    def subtract(self, reference, current):

        if reference is None:
            raise RuntimeError("Reference frame is not available.")
        if current is None:
            raise RuntimeError("Current frame is not available.")

        ref = reference.astype(np.int32)
        cur = current.astype(np.int32)

        valid = self.create_valid_mask(reference, current)

        ref_edges = self._detect_scene_edges(reference)
        cur_edges = self._detect_scene_edges(current)

        before = np.count_nonzero(valid)
        valid = valid & (~ref_edges) & (~cur_edges)

        if self.debug:
            after = np.count_nonzero(valid)
            removed_pct = 100 * (before - after) / max(before, 1)
            print(f"[DEBUG] Edge filter removed {before - after} px "
                  f"({removed_pct:.1f}%) from valid region")

        diff = ref - cur
        diff = np.clip(diff, 0, None)
        diff[~valid] = 0

        diff[diff > self.max_plausible_height_mm] = 0

        self.difference = diff.astype(np.uint16)
        return self.difference

    def threshold(self, diff):

        res_map = self.expected_resolution(self.reference)

        dynamic_threshold = np.maximum(
            self.min_height,
            self.threshold_safety_factor * res_map
        )

        mask = np.zeros(diff.shape, dtype=np.uint8)
        mask[diff >= dynamic_threshold] = 255

        valid_thr = dynamic_threshold[self.reference > 0]
        if self.debug and len(valid_thr) > 0:
            print(
                f"[Adaptive Threshold] range: "
                f"{valid_thr.min():.1f} - {valid_thr.max():.1f} mm "
                f"(mean {valid_thr.mean():.1f} mm)"
            )

        return mask

    def morphology(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)
        return mask

    def keep_largest_component(self, mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels <= 1:
            return mask

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        output = np.zeros_like(mask)
        output[labels == largest] = 255
        return output

    def fill_largest_contour(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return mask

        largest = max(contours, key=cv2.contourArea)

        output = np.zeros_like(mask)
        cv2.drawContours(output, [largest], -1, 255, thickness=cv2.FILLED)
        return output

    def process_raw_mask(self):
        
        if self.reference is None:
            raise RuntimeError("Reference frame is not available.")
        if self.current is None:
            raise RuntimeError("Current frame is not available.")

        reference = self.remove_noise(self.reference)
        current = self.remove_noise(self.current)

        diff = self.subtract(reference, current)

        mask = self.threshold(diff)
        if self.debug:
            print(f"[DEBUG] After threshold        : {np.count_nonzero(mask)} px")

        mask = self.morphology(mask)
        if self.debug:
            print(f"[DEBUG] After morphology        : {np.count_nonzero(mask)} px")

        mask = self.keep_largest_component(mask)
        if self.debug:
            print(f"[DEBUG] After keep_largest_comp : {np.count_nonzero(mask)} px")

        mask = self.fill_largest_contour(mask)
        if self.debug:
            print(f"[DEBUG] After fill_largest_cont : {np.count_nonzero(mask)} px")

        return diff, mask


    def visualize_depth(self, depth):
        normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    def visualize_difference(self, diff):
        normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)

    def diagnose_diff_quality(self, diff, mask, expected_height_mm=None):
     
        valid = diff[mask > 0]
        valid = valid[valid > 0]
        if len(valid) == 0:
            print("[Diff Quality] No valid pixels to evaluate.")
            return

        p25, p50, p75 = np.percentile(valid, [25, 50, 75])
        iqr = p75 - p25

        print(f"[Diff Quality] median={p50:.0f}mm, IQR={iqr:.0f}mm")

        if iqr > 0.5 * p50:
            print("Warning: diff distribution is spread out — "
                  "possible contamination from noise/background.")

        if expected_height_mm and abs(p50 - expected_height_mm) > 0.3 * expected_height_mm:
            print(f"Warning: median diff ({p50:.0f}mm) is far from the "
                  f"expected height ({expected_height_mm:.0f}mm).") 