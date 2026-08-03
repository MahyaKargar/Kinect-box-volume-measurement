import cv2
import traceback
import numpy as np
import time

from camera import KinectCamera
from processing import DepthProcessor
from measurement import PointCloudBuilder
from volume_calculator import VolumeCalculator


def wait_for_frame(camera, num_frames=15):

    print(f"Capturing {num_frames} depth frames...")

    frames = []

    while len(frames) < num_frames:

        frame = camera.get_depth_frame()

        if frame is None:
            continue

        frames.append(frame)

        time.sleep(0.03)

        print(
            f"\rFrame {len(frames)}/{num_frames}",
            end=""
        )

    print("\nCreating median frame...")

    frames = np.stack(frames, axis=0)

    median_frame = np.median(
        frames,
        axis=0
    )

    print("Stable frame captured.\n")

    return median_frame.astype(np.uint16)

def show_depth(camera, processor):

    print("\nLive Depth View")
    print("Press ESC to return to the menu.\n")

    while True:

        depth = camera.get_depth_frame()

        if depth is None:
            continue

        image = processor.visualize_depth(depth)

        cv2.imshow("Depth", image)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:

            cv2.destroyWindow("Depth")
            break


def capture_reference(camera, processor):

    depth = wait_for_frame(camera, num_frames=30)

    print("8191 Reference :", np.count_nonzero(depth == 8191))
    processor.set_reference(depth)

    print("Reference frame captured successfully.")


def capture_current(camera, processor):

    depth = wait_for_frame(camera, num_frames=30)

    print("8191 Reference :", np.count_nonzero(depth == 8191))

    processor.set_current(depth)

    print("Current frame captured successfully.")


def save_reference(processor):

    try:

        processor.save_reference()

        print("Reference frame saved successfully.")

    except Exception as ex:

        print(ex)


def load_reference(processor):

    try:

        processor.load_reference()

        print("Reference frame loaded successfully.")

    except Exception as ex:

        print(ex)


def calculate_difference(processor, measurement, volume):

    if processor.reference is None:

        print("Reference frame not available.")
        return

    if processor.current is None:

        print("Current frame not available.")
        return

    diff, mask = processor.process()

    diff_show = cv2.convertScaleAbs(
        diff,
        alpha=255.0 / max(diff.max(), 1)
    )

    cv2.imshow("Raw Difference", diff_show)
    cv2.waitKey(0)
    # ایcv2.destroyWindow("Raw Difference")

    valid = diff[diff > 0]

    if len(valid):
        print("Diff mean (valid):", np.mean(valid))
        print("Max :", np.max(valid))
        print("Diff median:", np.median(valid))
        print("Mean:", np.mean(valid))
        print("Diff std:", np.std(valid))
        print("P10 :", np.percentile(valid,10))
        print("P25 :", np.percentile(valid,25))
        print("P50 :", np.percentile(valid,50))
        print("P75 :", np.percentile(valid,75))
        print("P90 :", np.percentile(valid,90))

    print("Reference mean:", np.mean(processor.reference))
    print("Current mean  :", np.mean(processor.current))

    print("Reference min :", np.min(processor.reference))
    print("Current min   :", np.min(processor.current))

    print("Reference max :", np.max(processor.reference))
    print("Current max   :", np.max(processor.current))

    print("Mask Pixels:", np.count_nonzero(mask))

    reference_cloud, current_cloud, object_cloud = measurement.process(
    processor.reference,
    processor.current,
    mask
    )

    print("--------------------------------")
    print("Reference Cloud :", reference_cloud.shape)
    print("Current Cloud   :", current_cloud.shape)
    print("Object Cloud    :", object_cloud.shape)
    print("--------------------------------")

    measurement.visualize_point_cloud(
        object_cloud,
        "Object Point Cloud"
    )

    contours = processor.find_contours(mask)

    result = processor.visualize_difference(diff)

    cv2.drawContours(
        result,
        contours,
        -1,
        (255, 255, 255),
        2
    )

    print("--------------------------------")
    print(f"Contours detected : {len(contours)}")
    print("Press ESC to return.")
    print("--------------------------------")

    volume_cm3 = volume.calculate_volume_from_diff(diff, processor.reference, mask)
    width, length, height = volume.calculate_bounding_box(object_cloud, diff, mask)

    print(f"Volume: {volume_cm3:.2f} cm³")
    print(f"Dimensions: {width:.1f} x {length:.1f} x {height:.1f} cm")

    while True:

        cv2.imshow("Difference", result)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:

            cv2.destroyWindow("Difference")
            cv2.destroyWindow("Mask")

            break
def show_reference(processor):

    if not processor.has_reference():

        print("Reference frame is not available.")

        return

    print("Showing Reference Frame")
    print("Press ESC to return.")

    while True:

        image = processor.visualize_depth(
            processor.reference
        )

        cv2.imshow(
            "Reference Frame",
            image
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:

            cv2.destroyWindow(
                "Reference Frame"
            )

            break

def show_current(processor):

    if not processor.has_current():

        print("Current frame is not available.")

        return

    print("Showing Current Frame")
    print("Press ESC to return.")

    while True:

        image = processor.visualize_depth(
            processor.current
        )

        cv2.imshow(
            "Current Frame",
            image
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:

            cv2.destroyWindow(
                "Current Frame"
            )

            break

def print_menu():

    print()
    print("=" * 45)
    print("      Kinect Volume Measurement")
    print("=" * 45)

    print("1. Live Depth View")
    print("2. Capture Reference Frame")
    print("3. Show Reference Frame")
    print("4. Capture Current Frame")
    print("5. Show Current Frame")
    print("6. Calculate Difference")
    print("7. Save Reference Frame")
    print("8. Load Reference Frame")
    print("0. Exit")

    print("=" * 45)

def main():

    camera = None

    try:

        print("Initializing Kinect...")

        camera = KinectCamera()

        processor = DepthProcessor()

        measurement = PointCloudBuilder()

        volume = VolumeCalculator()

        camera.start()

        print("Kinect started successfully.")

        while True:

            print_menu()

            choice = input("Select an option: ").strip()

            if choice == "1":

                show_depth(camera, processor)

            elif choice == "2":

                capture_reference(camera, processor)

            elif choice == "3":

                show_reference(processor)

            elif choice == "4":

                capture_current(camera, processor)

            elif choice == "5":

                show_current(processor)

            elif choice == "6":

                calculate_difference(processor, measurement, volume)

            elif choice == "7":

                save_reference(processor)

            elif choice == "8":

                load_reference(processor)

            elif choice == "0":

                print("Exiting program...")

                break

            else:

                print("Invalid option.")

    except KeyboardInterrupt:

        print("\nProgram interrupted by user.")

    except Exception:

        traceback.print_exc()

    finally:

        print("\nStopping Kinect...")

        if camera is not None:

            try:

                camera.stop()

            except Exception:

                pass

        cv2.destroyAllWindows()

        print("Program closed successfully.")


if __name__ == "__main__":

    main() 