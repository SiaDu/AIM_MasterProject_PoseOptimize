import maya.cmds as cmds
import json
import os

# —— Defined export directory —— 
base_dir = "/home/s5722875/Notes/Semester3/SkeletonFinished/maya_default_pose_output/weight"
os.makedirs(base_dir, exist_ok=True)

# —— Ensure that the current pose is bound (optional)
# cmds.dagPose('-restore', 'bindPose1')

# 1. get all Joint
joints = cmds.ls(type='joint')
if not joints:
    raise RuntimeError("No Joint detected in the scene Joint")

# 2. Query the world matrix of each joint
bind_mats = {}
for j in joints:
    m = cmds.xform(j, q=True, ws=True, matrix=True)  # len: 16
    bind_mats[j] = [m[0:4], m[4:8], m[8:12], m[12:16]]

# 3. save JSON
out_path = os.path.join(base_dir, "bind_matrices.json")
with open(out_path, 'w') as f:
    json.dump(bind_mats, f, indent=2)

print(f“Exported {len(bind_mats)} global binding matrices of Joints to {out_path}”)
