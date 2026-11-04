import cv2
import json
import numpy as np
from pathlib import Path

# Configuration
IMAGE_DIR = Path("images/ArUco_images")
CALIBRATION_FILE = Path("camera_calibration.npz")
OUTPUT_FILE = Path("camera_poses.json")

MARKER_ID = 0
MARKER_SIZE = 0.136  # metres; width of the black square

# How the marker sits in your Godot scene (marker centre = Godot origin):
#   "flat" - lying on the floor/table. Marker normal = Godot +Y (up),
#            marker "up" edge points toward Godot -Z.
#   "wall" - standing upright, facing Godot +Z, marker "up" = Godot +Y.
MARKER_PLACEMENT = "flat"

MARKER_TO_GODOT_WORLD = {
    "flat": np.array([
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]),
    "wall": np.eye(3),
}[MARKER_PLACEMENT]

# OpenCV camera coordinates -> Godot camera coordinates.
# OpenCV: +X right, +Y down, +Z forward
# Godot:  +X right, +Y up,   -Z forward
CV_TO_GODOT_CAMERA = np.diag([1.0, -1.0, -1.0])

# Load calibration
calibration = np.load(CALIBRATION_FILE)
camera_matrix = calibration["camera_matrix"]
dist_coeffs = calibration["dist_coeffs"]

# Set up ArUco detector
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Marker corners in marker-local coordinates (X right, Y up, Z out of marker).
# Order: top-left, top-right, bottom-right, bottom-left.
half = MARKER_SIZE / 2
object_points = np.array(
    [
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ],
    dtype=np.float64,
)


# Convert a rotation matrix to Godot's YXZ Euler angles (Basis.get_euler()).
def matrix_to_godot_euler(basis):
    x = np.arcsin(np.clip(-basis[1, 2], -1.0, 1.0))
    cx = np.cos(x)

    if abs(cx) > 1e-6:
        y = np.arctan2(basis[0, 2], basis[2, 2])
        z = np.arctan2(basis[1, 0], basis[1, 1])
    else:
        # Gimbal-lock fallback.
        y = np.arctan2(-basis[2, 0], basis[0, 0])
        z = 0.0

    return np.degrees([x, y, z]).tolist()


# Supported image formats
extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
)

if not image_paths:
    raise FileNotFoundError(f"No images found in {IMAGE_DIR.resolve()}")

poses = []
failed = []

for image_path in image_paths:
    # IGNORE_ORIENTATION: don't auto-rotate by EXIF, so pixels match
    # the sensor layout the calibration was done on.
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )

    if image is None:
        print(f"Could not read: {image_path.name}")
        failed.append(image_path.name)
        continue

    height, width = image.shape[:2]

    if (width, height) != (4000, 3000):
        print(f"WARNING: {image_path.name} is {width}x{height}, not 4000x3000.")

    corners, ids, _ = detector.detectMarkers(image)

    if ids is None or MARKER_ID not in ids.flatten():
        print(f"No marker {MARKER_ID} found: {image_path.name}")
        failed.append(image_path.name)
        continue

    marker_index = int(np.where(ids.flatten() == MARKER_ID)[0][0])
    image_points = corners[marker_index].reshape(4, 2).astype(np.float64)

    # Marker -> OpenCV camera transform.
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not success:
        print(f"Pose estimation failed: {image_path.name}")
        failed.append(image_path.name)
        continue

    R_marker_to_cv, _ = cv2.Rodrigues(rvec)

    # Invert to get the camera pose in the marker frame.
    R_cv_to_marker = R_marker_to_cv.T
    camera_position_marker = -R_cv_to_marker @ tvec.reshape(3)

    # Godot-style camera axes expressed in marker coordinates.
    R_godot_camera_in_marker = R_cv_to_marker @ CV_TO_GODOT_CAMERA

    # Marker frame -> Godot world frame.
    position = (MARKER_TO_GODOT_WORLD @ camera_position_marker).tolist()
    R_world = MARKER_TO_GODOT_WORLD @ R_godot_camera_in_marker
    rotation_degrees = matrix_to_godot_euler(R_world)

    distance = float(np.linalg.norm(camera_position_marker))

    # Vertical FOV from calibration (Godot Camera3D default keep_aspect is
    # vertical). Uses the calibrated focal length scaled to this image size.
    fy = camera_matrix[1, 1]
    fov_v = float(np.degrees(2 * np.arctan(height / (2 * fy))))

    pose = {
        "image": image_path.name,
        "marker_id": MARKER_ID,
        "position": position,
        "rotation_degrees": rotation_degrees,
        "distance_m": distance,
        "fov_vertical_degrees": fov_v,
    }
    poses.append(pose)

    print(f"\n{image_path.name}")
    print("  Position (m): " + ", ".join(f"{v:.4f}" for v in position))
    print("  Rotation YXZ (degrees): " + ", ".join(f"{v:.2f}" for v in rotation_degrees))
    print(f"  Distance: {distance:.4f} m")
    print(f"  Vertical FOV: {fov_v:.2f} deg")

# Export successful poses.
OUTPUT_FILE.write_text(json.dumps(poses, indent=2), encoding="utf-8")

print(f"\nSaved {len(poses)} poses to {OUTPUT_FILE}")
print(f"Failed/skipped images: {len(failed)}")

for name in failed:
    print(f"  - {name}")
