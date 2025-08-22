import maya.cmds as cmds
import json
import os

# Select multiple meshes (Transform nodes)
meshes = cmds.ls(selection=True, type="transform")
out_dir = "/home/s5722875/Notes/Semester3/SkeletonFinished/maya_default_pose_output/weight"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for mesh in meshes:
    shape = cmds.listRelatives(mesh, shapes=True, type='mesh', fullPath=True)
    if not shape:
        continue
    shape = shape[0]
    vcount = cmds.polyEvaluate(shape, vertex=True)

    verts = []
    for i in range(vcount):
        vtx = f"{shape}.vtx[{i}]"
        pos = cmds.xform(vtx, query=True, translation=True, worldSpace=True)
        verts.append(pos)

    data = {
        "mesh": mesh,
        "vertex_count": vcount,
        "vertices": verts
    }
    filename = f"verts_rest_{mesh}.json"
    with open(os.path.join(out_dir, filename), 'w') as f:
        json.dump(data, f, indent=2)
    print(f"export {filename}")
