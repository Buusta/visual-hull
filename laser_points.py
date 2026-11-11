import cv2
import json
import numpy as np
from pathlib import Path
import time

# ---------------- Configuration ----------------
SCAN_DIR = Path("visual-hull-godot/laser_scan")      # copy of res://laser_scan (images + scan.json)
OUT_PLY = Path("visual-hull-godot/laser_points/laser_points.ply")
OUT_NPY = Path("visual-hull-godot/laser_points/laser_points.npy")
RED_THRESHOLD = 128                # red channel > threshold = laser pixel
MIN_RUN = 2                        # ignore runs shorter than this many pixels
MERGE_DIST = 0.001   # metres; points closer than this are merged (0 = off)
# ------------------------------------------------

t_start = time.perf_counter()

data = json.loads((SCAN_DIR / "scan.json").read_text())
plane_point = np.array(data["laser_plane"]["point"])
plane_normal = np.array(data["laser_plane"]["normal"])

cameras = []
for c in data["cameras"]:
    cameras.append({
        "origin": np.array(c["origin"]),
        "basis": np.stack([c["basis_x"], c["basis_y"], c["basis_z"]], axis=1),  # columns
        "fov": np.radians(c["fov_degrees"]),
        "keep_aspect": c["keep_aspect"],
    })

all_points = []

for step in data["steps"]:
    displacement = np.array(step["displacement"])

    for cam, image_name in zip(cameras, step["images"]):
        img = cv2.imread(str(SCAN_DIR / image_name), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not read {image_name}")
            continue
        h, w = img.shape[:2]

        mask = img[:, :, 2] > RED_THRESHOLD            # BGR: index 2 = red

        # Pad a dark row above and below, so every run has a start and an end.
        padded = np.zeros((h + 2, w), dtype=np.int8)
        padded[1:-1] = mask
        diff = np.diff(padded, axis=0)                 # +1 = run starts, -1 = run ends

        # Transposed, so both come out ordered by column, then by row.
        cols, starts = np.nonzero(diff.T == 1)
        _, ends = np.nonzero(diff.T == -1)             # first row after the run

        keep = (ends - starts) >= MIN_RUN
        if not keep.any():
            continue
        u = cols[keep] + 0.5                           # pixel-centre x
        v = (starts[keep] + ends[keep]) / 2            # run midpoint y

        # Pixel -> ray in Godot camera space (+X right, +Y up, -Z forward).
        if cam["keep_aspect"] == "height":
            tan_v = np.tan(cam["fov"] / 2)
            tan_h = tan_v * w / h
        else:
            tan_h = np.tan(cam["fov"] / 2)
            tan_v = tan_h * h / w
        x = (u / w * 2 - 1) * tan_h
        y = (1 - v / h * 2) * tan_v
        dirs_cam = np.stack([x, y, -np.ones_like(x)], axis=1)
        dirs = dirs_cam @ cam["basis"].T               # world-space ray directions

        # Ray-plane intersection.
        denom = dirs @ plane_normal
        t = ((plane_point - cam["origin"]) @ plane_normal) / np.where(np.abs(denom) < 1e-9, np.nan, denom)
        valid = np.isfinite(t) & (t > 0)
        points = cam["origin"] + dirs[valid] * t[valid, None]

        # Undo the target's movement so all slices share the original object frame.
        points -= displacement
        all_points.append(points)
        print(f"{image_name}: {len(points)} points")

if not all_points:
    raise RuntimeError("No laser pixels found. Check RED_THRESHOLD and the images.")

cloud = np.concatenate(all_points)

if MERGE_DIST > 0:
    before = len(cloud)
    cells = np.floor(cloud / MERGE_DIST).astype(np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    cloud = np.stack(
        [np.bincount(inverse, weights=cloud[:, k]) for k in range(3)], axis=1
    ) / counts[:, None]
    print(f"Merged {before} -> {len(cloud)} points")

np.save(OUT_NPY, cloud)

with open(OUT_PLY, "w") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(cloud)}\n")
    f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
    for p in cloud:
        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

print(f"\nSaved {len(cloud)} points to {OUT_PLY} and {OUT_NPY}")
print("Bounding box min:", cloud.min(axis=0), "max:", cloud.max(axis=0))

elapsed = time.perf_counter() - t_start
print(f"Processing time: {elapsed:.2f} s ({elapsed / len(data['steps']) * 1000:.1f} ms per step)")
