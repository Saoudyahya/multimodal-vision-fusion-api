import cv2
import time


def find_working_cameras():
    """Find all working camera indices"""
    print("\n" + "=" * 60)
    print("Searching for available cameras...")
    print("=" * 60 + "\n")

    working_cameras = []

    # Try indices 0-10 (covers most cases)
    for index in range(10):
        print(f"Testing camera index {index}...", end=" ")

        # Try DirectShow first (best for USB cameras on Windows)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if cap.isOpened():
            # Try to read a frame with timeout
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.5)  # Give camera time to initialize

            ret, frame = cap.read()

            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                print(f"✓ FOUND!")
                print(f"  Resolution: {width}x{height}")
                print(f"  FPS: {fps}")

                working_cameras.append({
                    'index': index,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'backend': 'DirectShow'
                })

                cap.release()
                continue

        cap.release()
        print("✗ Not available")

    print("\n" + "=" * 60)
    print(f"Found {len(working_cameras)} working camera(s)")
    print("=" * 60 + "\n")

    return working_cameras


def test_camera_capture(camera_index, duration=5):
    """Test capturing frames from a specific camera"""
    print(f"\n{'=' * 60}")
    print(f"Testing Camera {camera_index} for {duration} seconds")
    print(f"{'=' * 60}\n")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Failed to open camera!")
        return False

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Camera opened successfully!")
    print("Capturing frames...")

    start_time = time.time()
    frame_count = 0
    error_count = 0

    while time.time() - start_time < duration:
        ret, frame = cap.read()

        if ret and frame is not None:
            frame_count += 1
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"  Captured {frame_count} frames...")
        else:
            error_count += 1

        time.sleep(0.033)  # ~30 FPS

    cap.release()

    print(f"\n{'=' * 60}")
    print(f"Test Complete!")
    print(f"  Total frames: {frame_count}")
    print(f"  Errors: {error_count}")
    print(f"  FPS: {frame_count / duration:.2f}")
    print(f"{'=' * 60}\n")

    return frame_count > 0


def show_camera_preview(camera_index):
    """Show live preview from camera (press 'q' to quit)"""
    print(f"\n{'=' * 60}")
    print(f"Starting Camera {camera_index} Preview")
    print(f"Press 'q' to quit")
    print(f"{'=' * 60}\n")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Failed to open camera!")
        return False

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if ret and frame is not None:
            frame_count += 1

            # Add frame counter to image
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(f'Camera {camera_index} Preview', frame)
        else:
            print(f"Failed to read frame {frame_count}")

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nPreview stopped. Total frames: {frame_count}")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Camera Diagnostic Tool")
    print("=" * 60)

    # Find all working cameras
    cameras = find_working_cameras()

    if not cameras:
        print("\n❌ No working cameras found!")
        print("\nTroubleshooting tips:")
        print("1. Make sure your USB camera is properly connected")
        print("2. Try unplugging and replugging the camera")
        print("3. Check if camera works in Windows Camera app")
        print("4. Close any other apps using the camera (Skype, Teams, etc.)")
        input("\nPress Enter to exit...")
        exit(1)

    print("\nWorking cameras found:")
    for cam in cameras:
        print(f"  Index {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']} FPS")

    # Ask user which camera to test
    print("\n" + "=" * 60)
    if len(cameras) == 1:
        selected_index = cameras[0]['index']
        print(f"Using camera index: {selected_index}")
    else:
        while True:
            try:
                selected_index = int(input(f"Enter camera index to test (0-{len(cameras) - 1}): "))
                if selected_index in [cam['index'] for cam in cameras]:
                    break
                else:
                    print("Invalid index. Please try again.")
            except ValueError:
                print("Please enter a number.")

    # Test capturing frames
    test_camera_capture(selected_index, duration=3)

    # Ask if user wants to see preview
    show_preview = input("\nShow live preview? (y/n): ").lower().strip()
    if show_preview == 'y':
        show_camera_preview(selected_index)

    # Save the working camera index
    print(f"\n{'=' * 60}")
    print(f"✓ Camera index {selected_index} is working!")
    print(f"Use this index when starting your FastAPI server")
    print(f"Example: POST /camera/start?camera_index={selected_index}")
    print(f"{'=' * 60}\n")