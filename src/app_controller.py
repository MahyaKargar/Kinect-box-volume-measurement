import time
import cv2
import numpy as np

from camera import KinectCamera
from csv_logger import CSVLogger
from depth_processor import DepthProcessor
from object_analyzer import ObjectAnalyzer
from point_cloud_builder import PointCloudBuilder
from volume_calculator import VolumeCalculator


class AppController:

    def __init__(self):
        self.camera = KinectCamera()
        self.processor = DepthProcessor()
        self.analyzer = ObjectAnalyzer()
        self.volume = VolumeCalculator()
        self.point_cloud_builder = PointCloudBuilder()
        self.logger = CSVLogger()

    def start(self):
        print("Initializing Kinect...")
        self.camera.start()
        print("Kinect started successfully.")

    def stop(self):
        print("\nStopping Kinect...")
        if self.camera:
            try:
                self.camera.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("Program closed successfully.")


    def wait_for_frame(self, num_frames=15):
        print(f"Capturing {num_frames} depth frames...")
        frames = []
        while len(frames) < num_frames:
            frame = self.camera.get_depth_frame()
            if frame is None:
                continue
            frames.append(frame)
            time.sleep(0.03)
            print(f"\rFrame {len(frames)}/{num_frames}", end="")

        print("\nCreating median frame...")
        median_frame = np.median(np.stack(frames, axis=0), axis=0)
        print("Stable frame captured.\n")
        return median_frame.astype(np.uint16)


    def show_live_depth(self):
        print("\nLive Depth View - Press ESC to return.\n")
        while True:
            depth = self.camera.get_depth_frame()
            if depth is None:
                continue
            image = self.processor.visualize_depth(depth)
            cv2.imshow("Depth", image)
            if (cv2.waitKey(1) & 0xFF) == 27:
                cv2.destroyWindow("Depth")
                break

    def capture_reference(self):
        depth = self.wait_for_frame(num_frames=30)
        self.processor.set_reference(depth)
        print("Reference frame captured successfully.")

    def capture_current(self):
        depth = self.wait_for_frame(num_frames=30)
        self.processor.set_current(depth)
        print("Current frame captured successfully.")

    def show_reference(self):
        if not self.processor.has_reference():
            print("Reference frame is not available.")
            return
        self._show_frame_until_esc(
            "Reference Frame", self.processor.reference
        )

    def show_current(self):
        if not self.processor.has_current():
            print("Current frame is not available.")
            return
        self._show_frame_until_esc("Current Frame", self.processor.current)

    def save_reference(self):
        try:
            self.processor.save_reference()
            print("Reference frame saved successfully.")
        except Exception as ex:
            print(ex)

    def load_reference(self):
        try:
            self.processor.load_reference()
            print("Reference frame loaded successfully.")
        except Exception as ex:
            print(ex)

    def _show_frame_until_esc(self, title, frame_data):
        print(f"Showing {title} - Press ESC to return.")
        while True:
            image = self.processor.visualize_depth(frame_data)
            cv2.imshow(title, image)
            if (cv2.waitKey(1) & 0xFF) == 27:
                cv2.destroyWindow(title)
                break


    def calculate_difference(self):
        if not self.processor.has_reference():
            print("Reference frame not available.")
            return
        if not self.processor.has_current():
            print("Current frame not available.")
            return

        diff, mask = self.processor.process_raw_mask()
        mask = self.analyzer.filter_top_surface(diff, mask)

        diff_show = cv2.convertScaleAbs(
            diff, alpha=255.0 / max(diff.max(), 1)
        )
        cv2.imshow("Raw Difference", diff_show)
        cv2.waitKey(0)

        self._print_diff_stats(diff)
        self._print_frame_stats()

        mask_count = np.count_nonzero(mask)
        print("Mask Pixels:", mask_count)
        if mask_count < 500:
            print(
                f"Warning: only {mask_count} valid pixels detected. Adjust camera/object."
            )

        ref_cloud, cur_cloud, obj_cloud = self.point_cloud_builder.process(
            self.processor.reference, self.processor.current, mask
        )

        print("--------------------------------")
        print("Reference Cloud :", ref_cloud.shape)
        print("Current Cloud   :", cur_cloud.shape)
        print("Object Cloud    :", obj_cloud.shape)
        print("--------------------------------")

        self.point_cloud_builder.visualize_point_cloud(
            obj_cloud, "Object Point Cloud"
        )

        contours = self.analyzer.find_contours(mask)
        print("--------------------------------")
        print(f"Contours detected : {len(contours)}")
        print("Press ESC to return.")
        print("--------------------------------")

        dim_a, dim_b, angle_deg = self.analyzer.oriented_dimensions(
            mask, self.processor.reference
        )
        (
            width_old,
            length_old,
            height,
        ) = self.volume.calculate_bounding_box(obj_cloud, diff, mask)

        vol_pixel = self.volume.calculate_volume_from_diff(
            diff, self.processor.reference, mask
        )
        vol_prism = self.volume.calculate_volume_prism(dim_a, dim_b, height)
        (
            vol_footprint,
            area_cm2,
            height_trimmed,
        ) = self.volume.calculate_volume_footprint(
            mask, self.processor.reference, diff
        )
        height_centroid = self.volume.calculate_centroid_height(mask, diff)
        vol_centroid = area_cm2 * height_centroid

        # Print summaries
        print(f"Volume (1: Pixel Integration) : {vol_pixel:.2f} cm³")
        print(f"Volume (2: Prism)             : {vol_prism:.2f} cm³")
        print(
            f"Volume (3: Footprint - FINAL) : {vol_footprint:.2f} cm³ (Area={area_cm2:.1f} cm², H={height_trimmed:.1f} cm)"
        )
        print(f"Volume (4: Centroid)          : {vol_centroid:.2f} cm³")
        print(
            f"Dimensions (Oriented)        : {dim_a:.1f} x {dim_b:.1f} x {height:.1f} cm (Angle: {angle_deg:.1f}°)"
        )

        # Log results
        brightness = self.camera.get_average_brightness()
        invalid_pct = (
            100.0
            * np.count_nonzero(self.processor.current == 0)
            / self.processor.current.size
        )

        self.logger.log_light_result(
            brightness=brightness,
            invalid_pct=invalid_pct,
            noise_min_height_mm=self.processor.min_height,
            mask_pixels=int(mask_count),
            width_cm=dim_a,
            length_cm=dim_b,
            height_cm=height_trimmed,
            volume_cm3=vol_footprint,
        )

        # Show Output Windows
        result = self.processor.visualize_difference(diff)
        cv2.drawContours(result, contours, -1, (255, 255, 255), 2)
        oriented_view = self.analyzer.draw_oriented_box(result, mask)

        while True:
            cv2.imshow("Difference", oriented_view)
            cv2.imshow("Mask", mask)
            if (cv2.waitKey(1) & 0xFF) == 27:
                cv2.destroyWindow("Difference")
                cv2.destroyWindow("Mask")
                break

    def _print_diff_stats(self, diff):
        valid = diff[diff > 0]
        if len(valid) == 0:
            return

        print("Diff mean (valid):", np.mean(valid))
        print("Max :", np.max(valid))
        print("Diff median:", np.median(valid))
        print("Diff std:", np.std(valid))
        print("P10 :", np.percentile(valid, 10))
        print("P25 :", np.percentile(valid, 25))
        print("P50 :", np.percentile(valid, 50))
        print("P75 :", np.percentile(valid, 75))
        print("P90 :", np.percentile(valid, 90))

    def _print_frame_stats(self):
        print("Reference mean:", np.mean(self.processor.reference))
        print("Current mean  :", np.mean(self.processor.current))
        print("Reference min :", np.min(self.processor.reference))
        print("Current min   :", np.min(self.processor.current))
        print("Reference max :", np.max(self.processor.reference))
        print("Current max   :", np.max(self.processor.current))