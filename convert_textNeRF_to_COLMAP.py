import json
import os
import numpy as np

# Input and output paths
input_json = "data/textNeRF_meta/scene_0000/meta.json"
output_dir = "data/textNeRF/sparse/0"
os.makedirs(output_dir, exist_ok=True)

# Load meta.json
with open(input_json, "r") as f:
    meta = json.load(f)

# Write cameras.txt
# Assuming all images share the same intrinsics (adjust if they differ)
camera_id = 1
intrinsic = list(meta.values())[0]["intrinsic"]  # Take first entry as template
fx = intrinsic[0][0]  # Focal length x
fy = intrinsic[1][1]  # Focal length y
cx = intrinsic[0][2]  # Principal point x
cy = intrinsic[1][2]  # Principal point y
with open(os.path.join(output_dir, "cameras.txt"), "w") as f:
    # Format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # Using PINHOLE model: fx, fy, cx, cy
    width, height = list(meta.values())[0]["image_wh"]
    f.write(f"{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

# Write images.txt
with open(os.path.join(output_dir, "images.txt"), "w") as f:
    f.write("# Image list with two lines of data per image:\n")
    f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    
    image_id = 1
    for img_path, data in meta.items():
        pose = np.array(data["pose"])  # 3x4 matrix [R | t]
        R = pose[:, :3]  # Rotation matrix (3x3)
        t = pose[:, 3]   # Translation vector (3x1)

        # Convert rotation matrix to quaternion
        # Using a simple method (assumes well-formed rotation matrix)
        trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        # Translation
        tx, ty, tz = t

        # Write image entry
        img_name = os.path.basename(img_path)
        f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {img_name}\n")
        f.write("\n")  # No 2D-3D correspondences yet
        image_id += 1

# Placeholder for points3D.txt (empty initially)
with open(os.path.join(output_dir, "points3D.txt"), "w") as f:
    f.write("# Empty point cloud\n")