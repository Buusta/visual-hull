import cv2
import numpy as np

# Physical marker size, in metres.
MARKER_SIZE = 0.136

# Load camera calibration.
calibration = np.load("camera_calibration.npz")
camera_matrix = calibration["camera_matrix"]
dist_coeffs = calibration["dist_coeffs"]

# Load image.
image = cv2.imread("images/distance_calibration/d965mm.png")

if image is None:
    raise FileNotFoundError("Could not load the image.")

# Check that the image resolution matches the calibration resolution.
height, width = image.shape[:2]
print(f"Image resolution: {width} x {height}")

# Set up ArUco detector.
dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Detect markers.
corners, ids, rejected = detector.detectMarkers(image)

if ids is None:
    print("No ArUco markers detected.")
else:
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    # Define the four marker corners in 3D.
    # Their order must match the detected image corners:
    # top-left, top-right, bottom-right, bottom-left.
    half_size = MARKER_SIZE / 2

    object_points = np.array(
        [
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0],
        ],
        dtype=np.float32,
    )

    for i, marker_id in enumerate(ids.flatten()):
        image_points = corners[i].reshape(4, 2)

        # Estimate marker rotation and translation.
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if not success:
            print(f"Could not estimate pose for marker {marker_id}.")
            continue

        # Translation relative to the camera, in metres.
        x, y, z = tvec.flatten()
        distance = np.linalg.norm(tvec)

        print(f"\nMarker ID: {marker_id}")
        print(f"X: {x * 1000:.1f} mm")
        print(f"Y: {y * 1000:.1f} mm")
        print(f"Z: {z * 1000:.1f} mm")
        print(f"Straight-line distance: {distance * 1000:.1f} mm")

        # Draw the marker's coordinate axes.
        cv2.drawFrameAxes(
            image,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            MARKER_SIZE * 0.5,
        )

        # Display distance above the detected marker.
        marker_top_left = corners[i][0][0].astype(int)
        label_position = (
            int(marker_top_left[0]),
            max(30, int(marker_top_left[1]) - 15),
        )

        cv2.putText(
            image,
            f"{distance * 100:.1f} cm",
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Resize for comfortable viewing without changing pose calculations.
    display_image = cv2.resize(
        image,
        None,
        fx=0.5,
        fy=0.5,
    )

    cv2.imshow("ArUco Pose Estimation", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
