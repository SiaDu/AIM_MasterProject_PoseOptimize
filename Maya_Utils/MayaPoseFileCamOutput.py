"""
Export Camera
Instructions:
  1. Select the Camera Transform you want to export.
  2. Run the script.
"""
import maya.cmds as cmds
import maya.api.OpenMaya as om2
import json, os, math

FILE   = "test1"  # cam file
OUTPUT_DIR  = "/home/s5722875/Notes/Semester3/AIM_MasterProject_PoseOptimize/maya_posefile_output"
# --------------------------
suffix = "_cam"
FILE_NAME = FILE + suffix

# ---------- Selected camera ----------
sel = cmds.ls(selection=True, type="transform", long=True)
if not sel:
    cmds.error("请先选中 Camera Transform 再运行脚本")
cam_transform = sel[0]

# ---------- Camera information ----------
cm = cmds.xform(cam_transform, q=True, ws=True, m=True)
world_mat = [cm[0:4], cm[4:8], cm[8:12], cm[12:16]]
cam_shape = cmds.listRelatives(cam_transform, shapes=True, type="camera", fullPath=True)[0]

camera_info = {
    "world_matrix":    world_mat,
    "focal_length_mm": cmds.getAttr(f"{cam_shape}.focalLength"),
    "film_width_mm":   cmds.getAttr(f"{cam_shape}.horizontalFilmAperture") * 25.4,  # in → mm
    "film_height_mm":  cmds.getAttr(f"{cam_shape}.verticalFilmAperture")  * 25.4,
    "near_clip":       cmds.getAttr(f"{cam_shape}.nearClipPlane"),
    "far_clip":        cmds.getAttr(f"{cam_shape}.farClipPlane")
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, f"{FILE_NAME}.json")
with open(out_path, "w") as f:
    json.dump(camera_info, f, indent=2)

print(u"✓ export Camera →", out_path)
