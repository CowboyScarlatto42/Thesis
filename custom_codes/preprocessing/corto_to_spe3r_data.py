import os
import shutil
import re
import json
import numpy as np
import argparse


"""
Convert a CORTO-style rendered sequence into the intermediate SPE3R-like format.

The thesis preprocessing uses this script as the first bridge between raw CORTO
outputs and the NeuS input convention. It copies and renames RGB images and
masks, writes a camera intrinsic file, and derives `labels.json` from
`geometry.json`.

The most important part is the camera-frame conversion. CORTO/Blender cameras
use +X right, +Y up, and -Z forward, while the later OpenCV/NeuS projection path
uses +X right, +Y down, and +Z forward. The `blender_camera` frame fix applies
that conversion before saving target pose labels.
"""


def copy_and_rename_images(input_folder, output_folder):
    """Copy numerically named PNG images and rename them as img000001.png, ..."""
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    # Sort by the numeric stem used by the CORTO export.
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    for i, filename in enumerate(files, start=1):
        new_name = f"img{i:06d}.png"
        
        src = os.path.join(input_folder, filename)
        dst = os.path.join(output_folder, new_name)
        
        shutil.copy2(src, dst)

    print(f"Fatto: copiate {len(files)} immagini in {output_folder}")

def copy_and_rename_masks(input_folder, output_folder):
    """Copy CORTO mask files and align their order with the renamed images."""
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    def extract_index(filename):
        match = re.search(r"mask_(\d+)_\d+\.png", filename)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Nome file non valido: {filename}")

    # Sort using the frame index embedded in mask_<frame>_<id>.png.
    files.sort(key=extract_index)

    for i, filename in enumerate(files, start=1):
        new_name = f"img{i:06d}.png"

        src = os.path.join(input_folder, filename)
        dst = os.path.join(output_folder, new_name)

        shutil.copy2(src, dst)

    print(f"Fatto: copiate {len(files)} mask rinominate in {output_folder}")

def create_camera_json(
    output_folder,
    Nu,
    Nv,
    ppx,
    ppy,
    fx,
    fy,
    ccx,
    ccy,
    camera_matrix,
    dist_coeffs,
    filename="camera.json"
):
    """Write a SPE3R-like camera.json file with intrinsics and distortion."""
    os.makedirs(output_folder, exist_ok=True)

    camera_data = {
        "Nu": Nu,
        "Nv": Nv,
        "ppx": ppx,
        "ppy": ppy,
        "fx": fx,
        "fy": fy,
        "ccx": ccx,
        "ccy": ccy,
        "cameraMatrix": camera_matrix,
        "distCoeffs": dist_coeffs
    }

    output_path = os.path.join(output_folder, filename)

    with open(output_path, "w") as f:
        json.dump(camera_data, f, indent=2)

    print(f"Creato file: {output_path}")


def quat_normalize(q):
    """Normalize a quaternion represented as [w, x, y, z]."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Quaternione nullo.")
    return (q / n).tolist()


def quat_conjugate(q):
    """Return the conjugate of a [w, x, y, z] quaternion."""
    w, x, y, z = q
    return [w, -x, -y, -z]


def quat_multiply(q1, q2):
    """Hamilton product for quaternions in [w, x, y, z] order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]


def quat_rotate_vector(q, v):
    """Rotate vector `v` by quaternion `q`, both in the world convention."""
    q = quat_normalize(q)
    vq = [0.0, v[0], v[1], v[2]]
    q_conj = quat_conjugate(q)
    out = quat_multiply(quat_multiply(q, vq), q_conj)
    return out[1:]


def reorder_quaternion(q_wxyz, output_order="xyzw"):
    """Convert a quaternion from [w, x, y, z] to the requested output order."""
    w, x, y, z = q_wxyz

    if output_order == "wxyz":
        return [w, x, y, z]
    elif output_order == "xyzw":
        return [x, y, z, w]
    else:
        raise ValueError("output_order deve essere 'wxyz' o 'xyzw'")


def rotmat_to_quat_wxyz(R):
    """
    Convert a 3x3 rotation matrix to a normalized [w, x, y, z] quaternion.
    """
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return quat_normalize([w, x, y, z])


def get_camera_frame_fix(convention):
    """Return the rotation that maps the selected camera convention to CV axes."""
    if convention == "blender_camera":
        # Standard Blender camera:
        # +X right, +Y up, -Z forward -> CV (+X right, +Y down, +Z forward).
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=float)

    raise ValueError(f"Convenzione camera non supportata: {convention}")


def build_labels(cam_pos, cam_quat, body_pos, body_quat, output_order, cam_frame_fix):
    """Build per-frame target pose labels in the converted camera frame."""
    q_fix_wxyz = rotmat_to_quat_wxyz(cam_frame_fix)
    labels = []
    rel_positions = []

    for i in range(len(cam_pos)):
        p_c = np.asarray(cam_pos[i], dtype=float)
        q_cw = quat_normalize(cam_quat[i])   # camera -> world
        p_t = np.asarray(body_pos[i], dtype=float)
        q_tw = quat_normalize(body_quat[i])  # target -> world

        q_wc = quat_conjugate(q_cw)

        q_tc_wxyz = quat_multiply(q_wc, q_tw)
        q_tc_wxyz = quat_normalize(q_tc_wxyz)
        q_tc_wxyz = quat_multiply(q_fix_wxyz, q_tc_wxyz)
        q_tc_wxyz = quat_normalize(q_tc_wxyz)

        dt_w = (p_t - p_c).tolist()
        r_rel = quat_rotate_vector(q_wc, dt_w)
        r_rel = (cam_frame_fix @ np.asarray(r_rel, dtype=float)).tolist()
        rel_positions.append(r_rel)

        labels.append({
            "filename": f"img{i+1:06d}",
            "q_vbs2tango_true": reorder_quaternion(q_tc_wxyz, output_order),
            "r_Vo2To_vbs_true": r_rel,
        })

    return labels, np.asarray(rel_positions, dtype=float)


def summarize_rel_positions(rel_positions, convention_name):
    z = rel_positions[:, 2]
    d = np.linalg.norm(rel_positions, axis=1)
    positive = int(np.sum(z > 0.0))
    near_zero = int(np.sum(np.abs(z) < 1e-8))
    summary = {
        "convention": convention_name,
        "positive_z": positive,
        "near_zero_z": near_zero,
        "total": int(len(z)),
        "z_min": float(z.min()),
        "z_mean": float(z.mean()),
        "z_max": float(z.max()),
        "dist_min": float(d.min()),
        "dist_mean": float(d.mean()),
        "dist_max": float(d.max()),
    }
    return summary


def print_summary_block(summary):
    print(f"[{summary['convention']}] positive_z={summary['positive_z']}/{summary['total']}, "
          f"near_zero_z={summary['near_zero_z']}, "
          f"z(min/mean/max)=({summary['z_min']:.6f}, {summary['z_mean']:.6f}, {summary['z_max']:.6f}), "
          f"dist(mean)={summary['dist_mean']:.6f}")


def choose_best_convention(candidates):
    return max(
        candidates,
        key=lambda item: (
            item["summary"]["positive_z"],
            -item["summary"]["near_zero_z"],
            item["summary"]["z_mean"],
        ),
    )

def generate_labels_from_geometry(
    geometry_json_path,
    output_labels_path,
    output_order="wxyz",
    camera_frame="blender_camera",
):
    """
    Generate labels.json from CORTO geometry.json.

    Assumptions:
    - camera.orientation = q_camera_to_world [w, x, y, z]
    - body.orientation   = q_target_to_world [w, x, y, z]

    Output in the SPE3R/SPEED-like convention:
    - q_vbs2tango_true: target orientation in the camera frame;
    - r_Vo2To_vbs_true: target position in the converted camera frame.

    Supported `camera_frame` values:
    - blender_camera
    """

    with open(geometry_json_path, "r") as f:
        geometry = json.load(f)

    cam_pos = geometry["camera"]["position"]
    cam_quat = geometry["camera"]["orientation"]
    body_pos = geometry["body"]["position"]
    body_quat = geometry["body"]["orientation"]

    n = len(cam_pos)

    if not (len(cam_quat) == len(body_pos) == len(body_quat) == n):
        raise ValueError("Le liste in geometry.json non hanno la stessa lunghezza.")

    cam_frame_fix = get_camera_frame_fix(camera_frame)
    labels, rel_positions = build_labels(
        cam_pos=cam_pos,
        cam_quat=cam_quat,
        body_pos=body_pos,
        body_quat=body_quat,
        output_order=output_order,
        cam_frame_fix=cam_frame_fix,
    )
    summary = summarize_rel_positions(rel_positions, camera_frame)

    print("Diagnostica frame camera:")
    print_summary_block(summary)

    os.makedirs(os.path.dirname(output_labels_path) or ".", exist_ok=True)

    with open(output_labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Creato: {output_labels_path}")
    print(f"Numero frame: {n}")
    print(f"Quaternion output order: {output_order}")
    print(f"Camera frame selected: {camera_frame}")
    print_summary_block(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CORTO → SPE3R dataset pipeline")

    # Main input/output paths.
    parser.add_argument("--images_in", type=str, required=True)
    parser.add_argument("--masks_in", type=str, required=True)
    parser.add_argument("--geometry", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Camera parameters; these are usually set from the Colab notebook.
    parser.add_argument("--Nu", type=int, default=1024)
    parser.add_argument("--Nv", type=int, default=1024)
    parser.add_argument("--fx", type=float, default=2903.6963)
    parser.add_argument("--fy", type=float, default=2903.6963)
    parser.add_argument("--ccx", type=float, default=512)
    parser.add_argument("--ccy", type=float, default=512)
    parser.add_argument(
        "--camera_frame",
        type=str,
        default="blender_camera",
        choices=["blender_camera"],
        help="Camera-frame convention used when generating labels.json",
    )

    args = parser.parse_args()

    # === Output structure ===
    images_out = os.path.join(args.output_dir, "images")
    masks_out  = os.path.join(args.output_dir, "masks")

    # 1. Images.
    copy_and_rename_images(args.images_in, images_out)

    # 2. Masks.
    copy_and_rename_masks(args.masks_in, masks_out)

    # 3. camera.json.
    create_camera_json(
        output_folder=args.output_dir,
        Nu=args.Nu,
        Nv=args.Nv,
        ppx=1,
        ppy=1,
        fx=args.fx,
        fy=args.fy,
        ccx=args.ccx,
        ccy=args.ccy,
        camera_matrix=[
            [args.fx, 0, args.ccx],
            [0, args.fy, args.ccy],
            [0, 0, 1]
        ],
        dist_coeffs=[0, 0, 0, 0, 0]
    )

    # 4. labels.json.
    generate_labels_from_geometry(
        geometry_json_path=args.geometry,
        output_labels_path=os.path.join(args.output_dir, "labels.json"),
        output_order="wxyz",
        camera_frame=args.camera_frame,
    )

    print("\n✔ Pipeline completata.")
