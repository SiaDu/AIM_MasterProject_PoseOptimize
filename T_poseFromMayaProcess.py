import numpy as np
import json

# Module-level storage for skeletal data
parent_map = {}
sorted_entries = []
local_bind = {}
rotate_order = {}
depth_map = {}

def load_local_bind_and_orient(Tpose_json_path):
    """
    Read and set global parent_map, sorted_entries, local_bind, and rotate_order from joints_hierarchy_local.json 
        exported from Maya.
    Return the same data for functional use.
    """
    global parent_map, sorted_entries, local_bind, rotate_order

    # 1) read JSON
    with open(Tpose_json_path, 'r') as f:
        j_hierarchy = json.load(f)

    # 2) build parent_map
    parent_map_local = {}
    for entry in j_hierarchy:
        short = entry["name"].split("|")[-1]
        parent = entry.get("parent")
        parent_map_local[short] = parent.split("|")[-1] if parent else None

    # 3) Return to depth function
    depth_map_local = {}
    def get_depth(name):
        if parent_map_local[name] is None:
            return 0
        if name in depth_map_local:
            return depth_map_local[name]
        d = 1 + get_depth(parent_map_local[name])
        depth_map_local[name] = d
        return d

    # 4) Sort by depth
    sorted_entries_local = sorted(
        j_hierarchy,
        key=lambda e: get_depth(e["name"].split("|")[-1])
    )

    # 5) Construct local_bind and rotate_order
    local_bind_local   = {}
    rotate_order_local = {}
    for entry in sorted_entries_local:
        short = entry["name"].split("|")[-1]
        # Column-major order reading of a 4×4 homogeneous matrix
        local_bind_local[short] = np.array(entry.get("local_matrix", entry.get("world_matrix"))).reshape((4,4), order='F')
        rotate_order_local[short] = entry.get("rotateOrder", entry.get("rotate_order","xyz")).lower()

    # Assign to module-level variables
    parent_map     = parent_map_local
    sorted_entries = sorted_entries_local
    local_bind     = local_bind_local
    rotate_order   = rotate_order_local
    depth_map     = depth_map_local

    return parent_map, sorted_entries, local_bind, rotate_order, depth_map


def compute_bind_global(j):
    """
    Recursively calculate the global binding matrix of joint j, depending on the module-level 
        parent_map and local_bind.
    """
    L = local_bind[j]
    p = parent_map.get(j)
    if p is None:
        return L
    # 递归调用
    return compute_bind_global(p) @ L