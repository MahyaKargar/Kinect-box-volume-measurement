import os

import cv2
import numpy as np


class DepthProcessor:

    def __init__(self):

        # -----------------------------
        # Frames
        # -----------------------------

        self.reference = None
        self.current = None
        self.difference = None

        # -----------------------------
        # Parameters
        # -----------------------------

        self.min_height = 50

        self.kernel = np.ones(
            (5, 5),
            np.uint8
        )

    # ==================================================
    # Reference Frame
    # ==================================================

    def set_reference(self, depth):

        self.reference = depth.copy()

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

    # ==================================================
    # Current Frame
    # ==================================================

    def set_current(self, depth):

        self.current = depth.copy()

    def has_current(self):

        return self.current is not None

    # ==================================================
    # Difference
    # ==================================================

    def subtract(self):

        if self.reference is None:
            raise RuntimeError(
                "Reference frame is not available."
            )

        if self.current is None:
            raise RuntimeError(
                "Current frame is not available."
            )

        ref = self.reference.astype(np.int32)
        cur = self.current.astype(np.int32)

        diff = ref - cur

        diff = np.clip(diff, 0, None)

        self.difference = diff.astype(np.uint16)

        return self.difference

    # ==================================================
    # Invalid Pixels
    # ==================================================

    def remove_invalid_pixels(self, depth):

        depth = depth.copy()

        depth[(depth < 500)] = 0
        depth[(depth > 4000)] = 0

        return depth
    # ==================================================
    # Filters
    # ==================================================

    def median_filter(self, depth):

        return cv2.medianBlur(depth, 5)

    def gaussian_filter(self, depth):

        return cv2.GaussianBlur(
            depth,
            (5, 5),
            0
        )

    def remove_noise(self, depth):

        depth = self.remove_invalid_pixels(depth)

        depth = self.median_filter(depth)

        depth = self.gaussian_filter(depth)

        return depth

    # ==================================================
    # Threshold
    # ==================================================

    def threshold(self, diff):

        mask = np.zeros(
            diff.shape,
            dtype=np.uint8
        )

        mask[diff > self.min_height] = 255

        return mask

    # ==================================================
    # Morphology
    # ==================================================

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

    # ==================================================
    # Complete Pipeline
    # ==================================================

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

        ref = reference.astype(np.int32)

        cur = current.astype(np.int32)

        diff = ref - cur

        diff = np.clip(diff, 0, None)

        diff = diff.astype(np.uint16)

        self.difference = diff

        mask = self.threshold(diff)

        mask = self.morphology(mask)

        return diff, mask
    # ==================================================
    # Contours
    # ==================================================

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

    # ==================================================
    # Bounding Box
    # ==================================================

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

    # ==================================================
    # Draw
    # ==================================================

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

    # ==================================================
    # Visualization
    # ==================================================

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

    # ==================================================
    # Object Extraction
    # ==================================================

    def get_object_mask(self):

        diff, mask = self.process()

        contours = self.find_contours(mask)

        contour = self.largest_contour(contours)

        return diff, mask, contour

    def get_object_pixels(self):

        diff, mask, contour = self.get_object_mask()

        pixels = diff[mask > 0]

        return pixels

    # ==================================================
    # Statistics
    # ==================================================

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