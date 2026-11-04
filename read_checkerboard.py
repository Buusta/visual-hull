import cv2
import numpy as np
import glob
import os

# Calibration settings
CHECKERBOARD = (9, 6)  # Inner corners: columns, rows
SQUARE_SIZE = 0.017914  # Metres

IMAGE_DIR = "images/checkerboard_calibration"
OUTPUT_FILE = "camera_calibration.npz"

# Prepare the checkerboard's known 3D corner positions.
objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    dtype=np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

object_points = []
image_points = []

image_files = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)

if not image_files:
    raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

image_size = None
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

print(f"Found {len(image_files)} images.\n")

for filename in image_files:
    image = cv2.imread(filename)

    if image is None:
        print(f"Could not read: {filename}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = (gray.shape[1], gray.shape[0])
    elif image_size != (gray.shape[1], gray.shape[0]):
        print(f"Skipping different image resolution: {filename}")
        continue

    found, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD
    )

    if not found:
        print(f"Checkerboard NOT detected: {filename}")
        continue

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria
    )

    object_points.append(objp.copy())
    image_points.append(corners)

    print(f"Detected corners: {os.path.basename(filename)}")

print(f"\nSuccessful detections: {len(object_points)}/{len(image_files)}")

if len(object_points) < 8:
    raise RuntimeError(
        "Too few successful detections. Aim for at least 8 varied images."
    )

# Estimate camera intrinsics and lens distortion.
rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
    cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )
)

# Calculate per-image reprojection errors.
errors = []

for i, obj_points in enumerate(object_points):
    projected, _ = cv2.projectPoints(
        obj_points,
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    observed = image_points[i].reshape(-1, 2)
    projected = projected.reshape(-1, 2)

    point_errors = np.linalg.norm(
        observed - projected,
        axis=1
    )

    errors.append(point_errors.mean())

# Save calibration for later use.
np.savez(
    OUTPUT_FILE,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    image_size=np.array(image_size),
    square_size=SQUARE_SIZE,
    rms=rms
)

print("\n--- Calibration results ---")
print(f"Image resolution: {image_size[0]} x {image_size[1]}")
print(f"RMS reprojection error: {rms:.4f} pixels")
print(f"Mean per-image error: {np.mean(errors):.4f} pixels")
print(f"Camera matrix:\n{camera_matrix}")
print(f"Distortion coefficients:\n{dist_coeffs}")
print(f"\nSaved calibration to {OUTPUT_FILE}")
