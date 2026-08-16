import traceback
from app_controller import AppController


def print_menu():
    print("\n" + "=" * 45)
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
    app = AppController()

    try:
        app.start()

        actions = {
            "1": app.show_live_depth,
            "2": app.capture_reference,
            "3": app.show_reference,
            "4": app.capture_current,
            "5": app.show_current,
            "6": app.calculate_difference,
            "7": app.save_reference,
            "8": app.load_reference,
        }

        while True:
            print_menu()
            choice = input("Select an option: ").strip()

            if choice == "0":
                print("Exiting program...")
                break
            elif choice in actions:
                actions[choice]()
            else:
                print("Invalid option.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception:
        traceback.print_exc()
    finally:
        app.stop()


if __name__ == "__main__":
    main()