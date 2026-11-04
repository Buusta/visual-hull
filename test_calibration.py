import cv2
import numpy as np

calibration = np.load("camera_calibration.npz")

camera_matrix = calibration["camera_matrix"]
dist_coeffs = calibration["dist_coeffs"]

image = cv2.imread(
    "images/checkerboard_calibration/IMG_20261028_134720.png"
)

if image is None:
    raise FileNotFoundError("Could not load test image.")

height, width = image.shape[:2]

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    (width, height),
    1,
    (width, height)
)

undistorted = cv2.undistort(
    image,
    camera_matrix,
    dist_coeffs,
    None,
    new_camera_matrix
)

comparison = np.hstack((image, undistorted))

# Shrink for display.
scale = 0.5
comparison = cv2.resize(
    comparison,
    None,
    fx=scale,
    fy=scale
)

cv2.imshow("Original (left) | Undistorted (right)", comparison)

print("Press Q to close.")

while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
