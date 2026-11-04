import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from skimage import measure

# ---------------- Configuration ----------------
IMAGE_DIR = Path("images/ArUco_images")
MASK_DIR = Path("images/ArUco_masks")          # mask for IMG_x.png is IMG_x + MASK_SUFFIX + any extension
MASK_SUFFIX = "_Mask"
CALIBRATION_FILE = Path("camera_calibration.npz")
OUT_NPY = Path("voxels.npy")
OUT_OBJ = Path("hull.obj")

MARKER_ID = 0
MARKER_SIZE = 0.136
MARKER_PLACEMENT = "flat"                # must match export_poses.py

# Voxel grid, in Godot world coordinates (marker centre = origin).
GRID_CENTER = np.array([0.0, 0.25, 0.0])
GRID_SIZE = 0.50                          # metres per side
RES = 256                                # voxels per side

MIN_VIEWS = 2                            # voxel must be seen by at least this many cameras
MASK_THRESHOLD = 127                     # pixel > threshold = object (white), else background
SMOOTH_SIGMA = 0.6                       # blur before marching cubes; 0 = blocky
FLIP_FACES = True                       # set True if the mesh looks inside-out in Godot
# ------------------------------------------------

MARKER_TO_GODOT_WORLD = {
    "flat": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    "wall": np.eye(3),
}[MARKER_PLACEMENT]

calibration = np.load(CALIBRATION_FILE)
camera_matrix = calibration["camera_matrix"]
dist_coeffs = calibration["dist_coeffs"]

detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters(),
)
half = MARKER_SIZE / 2
object_points = np.array(
    [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
    dtype=np.float64,
)

# Voxel centres in Godot world, then in marker frame (M is orthonormal).
voxel = GRID_SIZE / RES
axis = (np.arange(RES) + 0.5) * voxel - GRID_SIZE / 2
gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
world_pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3) + GRID_CENTER
marker_pts = world_pts @ MARKER_TO_GODOT_WORLD      # == (M^T @ p) per point

occupied = np.ones(len(marker_pts), dtype=bool)
view_count = np.zeros(len(marker_pts), dtype=np.int32)

extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
image_paths = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in extensions)
used = 0

for image_path in image_paths:
    mask_candidates = list(MASK_DIR.glob(image_path.stem + MASK_SUFFIX + ".*"))
    if not mask_candidates:
        print(f"No mask for {image_path.name}, skipping")
        continue

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    mask = cv2.imread(str(mask_candidates[0]), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None or mask is None:
        print(f"Could not read {image_path.name} or its mask, skipping")
        continue

    h, w = image.shape[:2]
    if mask.shape != (h, w):
        print(f"{image_path.name}: resizing mask {mask.shape[::-1]} -> {(w, h)}")
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    solid = mask > MASK_THRESHOLD

    corners, ids, _ = detector.detectMarkers(image)
    if ids is None or MARKER_ID not in ids.flatten():
        print(f"No marker {MARKER_ID} in {image_path.name}, skipping")
        continue
    idx = int(np.where(ids.flatten() == MARKER_ID)[0][0])
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        corners[idx].reshape(4, 2).astype(np.float64),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        print(f"Pose failed for {image_path.name}, skipping")
        continue

    # Project all voxel centres (includes lens distortion).
    R, _ = cv2.Rodrigues(rvec)
    z_cam = marker_pts @ R[2] + tvec[2, 0]           # depth along OpenCV +Z
    uv, _ = cv2.projectPoints(marker_pts, rvec, tvec, camera_matrix, dist_coeffs)
    uv = uv.reshape(-1, 2)
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)

    visible = (z_cam > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    inside = np.zeros(len(marker_pts), dtype=bool)
    inside[visible] = solid[v[visible], u[visible]]

    # Carve only where this camera actually sees the voxel.
    occupied[visible & ~inside] = False
    view_count[visible] += 1
    used += 1
    print(f"{image_path.name}: carved, {occupied.sum()} voxels left")

if used == 0:
    raise RuntimeError("No cameras used.")

occupied &= view_count >= MIN_VIEWS
grid = occupied.reshape(RES, RES, RES)
print(f"\nUsed {used} cameras, {grid.sum()} occupied voxels")
np.save(OUT_NPY, grid)

if grid.sum() == 0:
    raise RuntimeError("Nothing left after carving; check masks, marker placement and grid position.")

# Marching cubes (pad so the surface closes at the grid border).
field = np.pad(grid.astype(np.float32), 1)
if SMOOTH_SIGMA > 0:
    field = gaussian_filter(field, SMOOTH_SIGMA)
verts, faces, normals, _ = measure.marching_cubes(field, level=0.5, spacing=(voxel,) * 3)

# Index space -> Godot world (padded by one voxel, voxel centres at +0.5).
grid_min = GRID_CENTER - GRID_SIZE / 2
verts = verts + grid_min - voxel * 0.5

if FLIP_FACES:
    faces = faces[:, ::-1]

with open(OUT_OBJ, "w") as f:
    for p in verts:
        f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    for t in faces + 1:
        f.write(f"f {t[0]} {t[1]} {t[2]}\n")

print(f"Saved {OUT_NPY} and {OUT_OBJ} ({len(verts)} verts, {len(faces)} tris)")
