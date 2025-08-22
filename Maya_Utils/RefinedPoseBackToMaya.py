# === Apply pose from JSON (rotateX/Y/Z only; no rotateOrder) ==================
import json
import maya.cmds as cmds

# --- config ---
POSE_JSON = "/home/s5722875/Notes/Semester3/AIM_MasterProject_PoseOptimize/refined_pose.json"
PREFIX    = ""           # if there is a namespace, write “charA:”; if there is a hierarchy, write “|rig|skeleton|”.
ROOT_NAME = "Root_M"     # Short name for root joint; if your project does not have this name, please change it.

# -----------------------------------------------------------------------------
def _node(name): return f"{PREFIX}{name}"

def _set_rotate_xyz(node, rx, ry, rz):
    # Unlock and set up rotateX/Y/Z（单位：度）
    for axis, val in zip(("X","Y","Z"), (rx, ry, rz)):
        attr = f"{node}.rotate{axis}"
        if cmds.objExists(attr):
            if cmds.getAttr(attr, lock=True):
                cmds.setAttr(attr, lock=False)
            cmds.setAttr(attr, val)

# -----------------------------------------------------------------------------
# 1) from JSON
with open(POSE_JSON, "r") as f:
    pose = json.load(f)

joint_rot = pose.get("joint_rot", {})
root_pos  = pose.get("root_pos", None)
root_rot  = pose.get("root_rot", None)
root_short = pose.get("root_name", ROOT_NAME)  
root_node  = _node(root_short)

# First set the root (world pose)
cmds.xform(root_node, ws=True, t=tuple(root_pos))
cmds.xform(root_node, ws=True, ro=tuple(root_rot))

# Write other joints in batches (skip root)
for j_short, eul in joint_rot.items():
    if j_short == root_short:     # ← skip
        continue
    node = _node(j_short)
    if cmds.objExists(node):
        rx, ry, rz = map(float, eul)
        cmds.setAttr(node + ".rotateX", rx)
        cmds.setAttr(node + ".rotateY", ry)
        cmds.setAttr(node + ".rotateZ", rz)
else:
    print("[apply_pose] All joints found.")

print("[apply_pose] Pose applied (rotateX/Y/Z only).")
