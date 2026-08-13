import os

import cv2
import numpy as np


class DepthProcessor:

    def __init__(self, fx=285.63, fy=285.63, baseline_mm=75.0, delta_disparity=0.125):
        self.reference = None
        self.current = None
        self.difference = None

        self.min_height = 20

        self.fx = fx
        self.fy = fy
        self.baseline_mm = baseline_mm
        self.delta_disparity = delta_disparity

        self.threshold_safety_factor = 2.5

        self.edge_gradient_threshold = 150.0
        self.top_surface_tolerance_mm = 50.0
        self.max_plausible_height_mm = 400.0
        self.max_min_height_ceiling_mm = 60.0

        self.kernel = np.ones(
            (5, 5),
            np.uint8
        )

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
    

    def set_reference(self, depth):
        self.reference = depth.copy()
        valid_pixels = self.reference[self.reference > 0]
        if len(valid_pixels) > 0:
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

            print(f"[Auto-Threshold] Base noise std (edge-filtered): {std_noise:.2f} mm")
            print(f"[Auto-Threshold] Scene edge pixels excluded: {edge_ratio * 100:.1f}%")
            print(f"[Auto-Threshold] Global min_height floor set to: {self.min_height:.2f} mm")
           
    

    def has_reference(self):
        return self.reference is not None

    def save_reference(self, filename="reference.npy"):

        if self.reference is None:
            raise RuntimeError(
                "Reference frame is not available."
            )

        np.save(filename, self.reference)

    def load_reference(self, filename="reference.npy"):

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        self.reference = np.load(filename)
        self.set_reference(self.reference)

  
    def set_current(self, depth):
        self.current = depth.copy()

    def has_current(self):
        return self.current is not None

    def subtract(self, reference, current):

        if reference is None:
            raise RuntimeError(
                "Reference frame is not available."
            )

        if current is None:
            raise RuntimeError(
                "Current frame is not available."
            )

        ref = reference.astype(np.int32)
        cur = current.astype(np.int32)

        valid = self.create_valid_mask(reference, current)

        # حذف پیکسل‌های نویزی لبه (flying pixels) در هر دو فریم
        ref_edges = self._detect_scene_edges(reference)
        cur_edges = self._detect_scene_edges(current)
        before_edge_filter = np.count_nonzero(valid)
        valid = valid & (~ref_edges) & (~cur_edges)
        after_edge_filter = np.count_nonzero(valid)
        print(f"[DEBUG] Edge filter removed {before_edge_filter - after_edge_filter} px "
              f"({100*(before_edge_filter-after_edge_filter)/max(before_edge_filter,1):.1f}%) "
              f"from valid region")

        diff = ref - cur

        diff = np.clip(diff, 0, None)
        diff[~valid] = 0

        diff[diff > self.max_plausible_height_mm] = 0

        self.difference = diff.astype(np.uint16)

        return self.difference

   

    # def remove_invalid_pixels(self, depth):

    #     depth = depth.copy()
    #     invalid = (depth == 0)
    #     depth[invalid] = 0
    #     return depth
    
    def median_filter(self, depth):
        return cv2.medianBlur(depth, 5)

    def gaussian_filter(self, depth):
        return cv2.GaussianBlur(
            depth,
            (5, 5),
            0
        )

    def remove_noise(self, depth):

        depth_clean = depth.copy()
        zero_mask = (depth_clean == 0)
        depth_clean = self.median_filter(depth_clean)
        depth_clean[zero_mask] = 0

        return depth_clean


    def threshold(self, diff):

        # valid = diff[diff > 0]

        # threshold = self.min_height

        res_map = self.expected_resolution(self.reference)

        dynamic_threshold = np.maximum(
            self.min_height,
            self.threshold_safety_factor * res_map
        )

        mask = np.zeros(
            diff.shape,
            dtype=np.uint8
        )

        # if len(valid) == 0:
        #     return mask

        # mean = np.mean(valid)
        # std = np.std(valid)

        # threshold = max(
        #     self.min_height,
        #     mean + 0.5 * std
        # )

        mask[diff >= dynamic_threshold] = 255

        valid_thr = dynamic_threshold[self.reference > 0]
        if len(valid_thr) > 0:
            print(
                f"[Adaptive Threshold] range: "
                f"{valid_thr.min():.1f} - {valid_thr.max():.1f} mm "
                f"(mean {valid_thr.mean():.1f} mm)"
            )


        # print(f"Adaptive Threshold : {threshold:.1f}")

        # mask[diff >= threshold] = 255

        return mask

   
    def morphology(self, mask):

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel
        )

        close_kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel
        )

        return mask

    def filter_top_surface(self, diff, mask):

           object_diff = diff[mask > 0]

           if len(object_diff) == 0:
               return mask

           median = np.median(object_diff)
           mad = np.median(np.abs(object_diff - median)) + 1e-6

           refined = np.zeros_like(mask)
           keep = (mask > 0) & (np.abs(diff.astype(np.float32) - median) <= 4  * 1.4826 * mad)
           refined[keep] = 255

           return refined
    def oriented_dimensions(self, mask, reference_depth):

        small_kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.erode(mask, small_kernel, iterations=1)

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


    def process(self):

        if self.reference is None:
            raise RuntimeError(
                "Reference frame is not available."
            )

        if self.current is None:
            raise RuntimeError(
                "Current frame is not available."
            )

        reference = self.remove_noise(self.reference)

        current = self.remove_noise(self.current)

        diff = self.subtract(reference, current)

        self.difference = diff

        mask = self.threshold(diff)
        print(f"[DEBUG] After threshold        : {np.count_nonzero(mask)} px")

        mask = self.morphology(mask)
        print(f"[DEBUG] After morphology        : {np.count_nonzero(mask)} px")

        mask = self.keep_largest_component(mask)
        print(f"[DEBUG] After keep_largest_comp : {np.count_nonzero(mask)} px")

        mask = self.fill_largest_contour(mask)
        print(f"[DEBUG] After fill_largest_cont : {np.count_nonzero(mask)} px")

        mask = self.filter_top_surface(diff, mask)
        print(f"[DEBUG] After filter_top_surface: {np.count_nonzero(mask)} px")

        return diff, mask
    
    def find_contours(self, mask):

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        return contours

    def largest_contour(self, contours):

        if len(contours) == 0:
            return None

        return max(
            contours,
            key=cv2.contourArea
        )

    def contour_area(self, contour):

        if contour is None:
            return 0

        return cv2.contourArea(contour)

   
    def bounding_box(self, contour):

        if contour is None:
            return None

        return cv2.boundingRect(contour)

    def object_center(self, contour):

        if contour is None:
            return None

        M = cv2.moments(contour)

        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return (cx, cy)

   
    def draw_contours(self, image, contours):

        output = image.copy()

        cv2.drawContours(
            output,
            contours,
            -1,
            (255, 255, 255),
            2
        )

        return output

    def draw_bounding_box(self, image, contour):

        if contour is None:
            return image

        output = image.copy()

        x, y, w, h = self.bounding_box(contour)

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        return output

    def draw_center(self, image, contour):

        if contour is None:
            return image

        output = image.copy()

        center = self.object_center(contour)

        if center is not None:

            cv2.circle(
                output,
                center,
                5,
                (0, 0, 255),
                -1
            )

        return output


    def visualize_depth(self, depth):

        normalized = cv2.normalize(
            depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        normalized = normalized.astype(np.uint8)

        return cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_JET
        )

    def visualize_difference(self, diff):

        normalized = cv2.normalize(
            diff,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        normalized = normalized.astype(np.uint8)

        return cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_HOT
        )

   
    def get_object_mask(self):

        diff, mask = self.process()

        contours = self.find_contours(mask)

        contour = self.largest_contour(contours)

        return diff, mask, contour

    def get_object_pixels(self):

        diff, mask, contour = self.get_object_mask()

        pixels = diff[mask > 0]

        return pixels

   
    def max_height(self):

        pixels = self.get_object_pixels()

        if len(pixels) == 0:
            return 0

        return float(np.max(pixels))

    def mean_height(self):

        pixels = self.get_object_pixels()

        if len(pixels) == 0:
            return 0

        return float(np.mean(pixels))

    def object_area_pixels(self):

        _, _, contour = self.get_object_mask()

        if contour is None:
            return 0

        return cv2.contourArea(contour)

    def create_valid_mask(self, reference, current):

        valid = (
            (reference > 0) &
            (current > 0)
        )

        return valid

    def keep_largest_component(self, mask):

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )

        if num_labels <= 1:
            return mask

        largest = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        output = np.zeros_like(mask)

        output[labels == largest] = 255

        return output


    def fill_largest_contour(self, mask):

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return mask

        largest = max(
            contours,
            key=cv2.contourArea
        )

        output = np.zeros_like(mask)

        cv2.drawContours(
            output,
            [largest],
            -1,
            255,
            thickness=cv2.FILLED
        )

        return output