import maya.cmds as cmds
import json, os

OUTPUT_DIR = "/home/s5722875/Notes/Semester3/AIM_MasterProject_PoseOptimize/maya_posefile_output"
POSE_NAME  = "C2P1S"
# --------------------------

# ---------- Get joint list ----------
joints_long = cmds.ls(type="joint", long=True)
if not joints_long:
    cmds.error("no joint in the scene")

# ---------- Identify root joints (parentless joints) ----------
root_long = next(j for j in joints_long
                 if not cmds.listRelatives(j, parent=True, type="joint", fullPath=True))
root_short = root_long.split("|")[-1]

# ---------- World position of root node + world rotation ----------
root_pos = cmds.xform(root_long, q=True, ws=True, t=True)      # cm
root_rot = cmds.xform(root_long, q=True, ws=True, ro=True)     # World Space Euler Angles

# ---------- Rotation values (rotateX/Y/Z) for the Channel Box of the remaining joints ----------
joint_rot = {}
for j_long in joints_long:
    short = j_long.split("|")[-1]
    if short == root_short:
        continue
    # Directly read the rotate attribute of Channel Box, returning (rx, ry, rz)
    rx, ry, rz = cmds.getAttr(j_long + ".rotate")[0]
    joint_rot[short] = [round(rx, 6), round(ry, 6), round(rz, 6)]

# Treat the root node as a joint and add a rotation channel with values of 0,0,0.
joint_rot[root_short] = [0.0, 0.0, 0.0]

# ---------- JSON ----------
pose_data = {
    "root_pos":  [round(x, 6) for x in root_pos],
    "root_rot":  [round(r, 6) for r in root_rot],
    "joint_rot": joint_rot
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, f"{POSE_NAME}.json")
with open(out_path, "w") as f:
    json.dump(pose_data, f, indent=2)

print(u"✓ exported ", out_path)
