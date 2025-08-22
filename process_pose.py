import numpy as np
import json
import os, glob
import copy
import transforms3d.euler as euler
from transforms3d.affines import compose


def build_pose_globals(
    sorted_entries,
    local_bind,
    parent_map,
    pose_JR,
    root_name="Root_M"
):
    """
    Generate the global transformation matrix for each joint under the specified pose.

    Parameters:
      sorted_entries: List of skeleton entries sorted by depth, each containing the key “name”.
      local_bind: dict, 4×4 local binding homogeneous matrix corresponding to each short_name.
      parent_map: dictionary, containing the parent node short_name or None for each short_name.
      pose_JR: dictionary, pose data containing “joint_rot”, ‘root_pos’, and “root_rot”.
      root_name: string, root node short_name.

    Returns:
      world_globals: dict, a 4×4 world homogeneous matrix corresponding to each short_name.
    """

    # 1) Root node world_mats (considering both base world_rot and joint_rot delta)
    world_globals = {}
    # ---- Base world translation + extrinsic world_rot ----
    root_pos     = np.array(pose_JR.get("root_pos", [0,0,0]))
    base_rot_deg = pose_JR.get("root_rot",   [0,0,0])
    base_rot_rad = np.radians(base_rot_deg)
    R_root_base  = euler.euler2mat(*base_rot_rad, axes='sxyz')
    # ---- Incremental local intrinsic rotation (from joint_rot) ----
    delta_deg    = pose_JR.get("joint_rot", {}).get(root_name, [0.0,0.0,0.0])
    delta_rad    = np.radians(delta_deg)
    # Default XYZ Intrinsic
    R_root_delta = euler.euler2mat(*delta_rad, axes='rxyz')
    # ---- Synthesize the final root node world rotation ----
    R_root = R_root_base @ R_root_delta
    world_globals[root_name] = compose(root_pos, R_root, [1,1,1])

    # 2) Treat the remaining joints in order.
    for entry in sorted_entries:
        name = entry["name"].split("|")[-1]
        if name == root_name:
            continue

        parent = parent_map.get(name) or root_name

        # local bind
        B = local_bind[name]
        t_bind = B[:3, 3]
        R_bind = B[:3, :3]

        # Pose rotation: Take the rotate channel (degrees) → Radians
        eul_deg = pose_JR.get("joint_rot", {}).get(name, [0.0, 0.0, 0.0])
        eul_rad = np.radians(eul_deg)
        order   = pose_JR.get("rotate_order_map", {}).get(name, "xyz")
        # Intrinsic rotation matrix
        R_p     = euler.euler2mat(*eul_rad, axes='r'+order)

        # Composite local transformation
        R_new   = R_bind @ R_p
        local_P = np.eye(4)
        local_P[:3, :3] = R_new
        local_P[:3, 3]  = t_bind

        # Overlay on parent node
        world_globals[name] = world_globals[parent] @ local_P

    return world_globals

def compute_camera_from_json(json_path, image_width, image_height):
    """
    Read the focal length and film dimensions from the camera JSON file exported from Blender/Maya, 
    and calculate the camera intrinsic matrix K.

    Args:
        json_path (str): Path to the camera parameter JSON file, containing the following fields:
            - focal_length_mm
            - film_width_mm
            - film_height_mm
            - world_matrix
        image_width (int): Horizontal resolution (pixels) of the rendered or displayed image 960
        image_height (int): Vertical resolution (pixels) of the rendered or displayed image 540

    Returns:
        numpy.ndarray: 3x3 camera intrinsic matrix K
        Transposed homogeneous matrix 4x4 camera world rotation & position cam_global
    """
    # read JSON
    with open(json_path, 'r') as f:
        cam = json.load(f)

    f_mm = cam['focal_length_mm']
    film_w = cam['film_width_mm']
    film_h = cam['film_height_mm']

    # Calculate pixel-level focal length
    fx = (f_mm / film_w) * image_width
    fy = (f_mm / film_h) * image_height

    # Main point assumption at the center of the image
    cx = image_width / 2.0
    cy = image_height / 2.0

    # Constructing the K matrix
    K = np.array([
        [fx,  0., cx],
        [0.,  fy, cy],
        [0.,  0., 1. ]
    ])
    
    # Transposed homogeneous 4x4 camera-to-world matrix cam_global
    M_cam_to_world = np.array(cam["world_matrix"], dtype = float).T

    return K, M_cam_to_world

def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return obj

def apply_offset(best_vec, LoAfix_poseJR, OPT_JOINTS, joint_idx_map, outfile=None):
    # 1) Rebuild the full pose_JR with your best_vec
    pose_JR_refined = copy.deepcopy(LoAfix_poseJR)
    joint_order=OPT_JOINTS
    
    for jn in joint_order:
        offset = joint_idx_map[jn]
        dx, dy, dz = best_vec[offset:offset+3]
        ox, oy, oz = pose_JR_refined["joint_rot"][jn]
        pose_JR_refined["joint_rot"][jn] = [ox+dx, oy+dy, oz+dz]
    
    if outfile is not None:
        with open(outfile, "w") as f:
            json.dump(to_jsonable(pose_JR_refined), f, indent=2, ensure_ascii=False)
        print("saved:", outfile)

    return pose_JR_refined


def load_skinning_data(weight_dir):
    """
    Read verts_rest_*.json / influences_*.json / weights_*.json /
    bind_global_inv.json and organize into several dictionaries.
    Return, influences_dict, all_weights, B_inv
    """

    # -------- A. rest pose verts --------
    verts_rest_dict = {}
    for path in glob.glob(os.path.join(weight_dir, "verts_rest_*.json")):
        with open(path, 'r') as f:
            d = json.load(f)
        verts_rest_dict[d["mesh"]] = {
            "vertex_count": d["vertex_count"],
            "vertices": np.asarray(d["vertices"], dtype=np.float32)
        }

    # -------- B. influences（joint sequence） --------
    influences_dict = {}
    for path in glob.glob(os.path.join(weight_dir, "influences_*.json")):
        with open(path, 'r') as f:
            d = json.load(f)
        influences_dict[d["mesh"]] = {
            "joints": d["influences"],
            "joint_count": d["joint_count"],
            "skinCluster": d["skinCluster"]
        }

    # -------- C. weights --------
    all_weights = {}
    for path in glob.glob(os.path.join(weight_dir, "weights_*.json")):
        d = json.load(open(path))
        all_weights[d["mesh"]] = np.asarray(d["weights"], dtype=np.float32)

    # -------- D. bind inverse matrix --------
    bind_path = os.path.join(weight_dir, "bind_matrices.json")
    bind_row  = json.load(open(bind_path))
    bind_inv_row = {j: np.linalg.inv(np.array(bind_row[j])) for j in bind_row}

    # → Row Major → 4×4；
    B_inv = {j: np.asarray(m, dtype=np.float32).reshape(4, 4).T
             for j, m in bind_inv_row.items()}

    return verts_rest_dict, influences_dict, all_weights, B_inv


# ------------------------------------------------------------
# 1) Single grid skin
# ------------------------------------------------------------
def skin_mesh(mesh, T_dict, verts_rest_dict, influences_dict,
              all_weights, B_inv):
    """
    Return the deformed vertex (V,3) of the specified mesh.
    """
    V_rest = verts_rest_dict[mesh]["vertices"]                 # (V,3)
    W      = all_weights[mesh]                                 # (V,J)
    joints = influences_dict[mesh]["joints"]                   # order ↔ W

    # Skin matrix for each joint (J,4,4)
    skin = np.stack([T_dict[j] @ B_inv[j] for j in joints], axis=0)

    # homogeneous coordinates
    V_h  = np.hstack([V_rest, np.ones((len(V_rest), 1), dtype=np.float32)])  # (V,4)
    VS   = (W @ skin.reshape(len(joints), 16)).reshape(-1, 4, 4) @ V_h[..., None]
    V_def = VS[:, :3, 0] / VS[:, 3, 0][:, None]               # divide w
    return V_def


# ------------------------------------------------------------
# 2) One-click skinning for the entire character
# ------------------------------------------------------------
# process_pose.py

def skin_character(weight_dir, T_dict, return_weights=False, weight_thr=1e-6):
    verts_rest_dict, influences_dict, all_weights, B_inv = load_skinning_data(weight_dir)

    # recommend to fix the mesh order to ensure alignment with per_vertex_weights.
    mesh_names = sorted(verts_rest_dict.keys())

    V_all_dict = {}
    per_vertex_weights = []  # list[dict], Same length as V_concat

    for mesh in mesh_names:
        V_all_dict[mesh] = skin_mesh(mesh, T_dict,
                                     verts_rest_dict, influences_dict,
                                     all_weights, B_inv)

        if return_weights:
            W      = all_weights[mesh]                          # (V,J)
            joints = influences_dict[mesh]["joints"]            # Corresponding to the column
            for row in W:
                wdict = {j: float(w) for j, w in zip(joints, row) if w > weight_thr}
                per_vertex_weights.append(wdict)

    V_concat = np.concatenate([V_all_dict[m] for m in mesh_names], axis=0)

    if return_weights:
        return V_all_dict, V_concat, per_vertex_weights
    return V_all_dict, V_concat



