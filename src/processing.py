import os

import cv2
import numpy as np


class DepthProcessor:

    def __init__(self):
        self.reference = None
        self.current = None
        self.difference = None

        self.min_height = 20

        self.kernel = np.ones(
            (5, 5),
            np.uint8
        )

    def set_reference(self, depth):
        self.reference = depth.copy()
        valid_pixels = self.reference[self.reference > 0]
        if len(valid_pixels) > 0:
            blur = cv2.medianBlur(self.reference, 5)
            noise_map = cv2.absdiff(self.reference, blur)
            valid_noise = noise_map[self.reference > 0]
            std_noise = np.std(valid_noise)
            self.min_height = max(20.0, 3.0 * std_noise)
            print(f"[Auto-Threshold] Dynamic noise threshold set to: {self.min_height:.2f} mm")

    

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

        diff = ref - cur

        diff = np.clip(diff, 0, None)
        diff[~valid] = 0

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

        threshold = self.min_height

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

        print(f"Adaptive Threshold : {threshold:.1f}")

        mask[diff >= threshold] = 255

        return mask

   
    def morphology(self, mask):

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.kernel
        )

        return mask

   

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

        mask = self.morphology(mask)

        mask = self.keep_largest_component(mask)

        mask = self.fill_largest_contour(mask)

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