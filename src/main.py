import cv2
import traceback

from camera import KinectCamera
from processing import DepthProcessor


def wait_for_frame(camera):

    print("Waiting for depth frame...")

    while True:

        frame = camera.get_depth_frame()

        if frame is not None:
            return frame


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

    depth = wait_for_frame(camera)

    processor.set_reference(depth)

    print("Reference frame captured successfully.")


def capture_current(camera, processor):

    depth = wait_for_frame(camera)

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


def calculate_difference(processor):

    if processor.reference is None:

        print("Reference frame not available.")
        return

    if processor.current is None:

        print("Current frame not available.")
        return

    diff = processor.subtract()

    diff = processor.remove_noise(diff)

    mask = processor.threshold(diff)

    mask = processor.morphology(mask)

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

                calculate_difference(processor)

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