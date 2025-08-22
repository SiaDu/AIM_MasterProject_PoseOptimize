import maya.cmds as cmds
import json
import os

OUTPUT_DIR = "/home/s5722875/Notes/Semester3/SkeletonFinished/maya_default_pose_output"

# —— Make sure you are in Bind Pose ——
# cmds.dagPose(‘-restore’, ‘<your BindPose node>’)

# Table corresponding to the rotateOrder of joints
ROT_ORDERS = ["xyz","yzx","zxy","xzy","yxz","zyx"]

# 1. Collect all Joints (long names)
all_joints = cmds.ls(type='joint')
joints = cmds.ls(all_joints, long=True)

out = []
for j in joints:
    # parent Joint (None if not applicable)
    parent = cmds.listRelatives(j, parent=True, type='joint')
    parent_name = parent[0] if parent else None

    # Current Joint's 4×4 local binding matrix (Bind Pose) in parent space
    local_m = cmds.xform(j, q=True, os=True, m=True)

    # get rotateOrder
    ro_idx = cmds.getAttr(j + ".rotateOrder")       # int 0–5
    ro_str = ROT_ORDERS[ro_idx]

    out.append({
        "name": j,
        "parent": parent_name,
        "local_matrix": local_m,   # row-major list of 16
        "rotateOrder": ro_str
    })

# 2. save
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "joints_hierarchy_local.json")
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)

print(f"Exported local bind matrices to: {out_path}")