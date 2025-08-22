import maya.cmds as cmds
import json
import os

# —— 1. Get all target meshes —— 
# If you want to process all meshes in the scene, you can use:
meshes = cmds.ls(type=‘mesh’, long=True)
# If you only want to process the currently selected ones, you can change it to:
# meshes = cmds.ls(selection=True, dag=True, shapes=True, type=‘mesh’, long=True)

if not meshes:
    raise RuntimeError("No Mesh detected in the scene (or select the Mesh to be processed first)")

# —— 2. Prepare to export the directory —— 
base_dir = "/home/s5722875/Notes/Semester3/SkeletonFinished/maya_default_pose_output/weight"
os.makedirs(base_dir, exist_ok=True)

for mesh_shape in meshes:
    # Derive transform name from shape name (optional, used for file naming)
    transform = cmds.listRelatives(mesh_shape, parent=True, fullPath=False)[0]

    # Find the skinCluster connected to this Mesh.
    hist = cmds.listHistory(mesh_shape) or []
    skin_clusters = cmds.ls(hist, type='skinCluster')
    if not skin_clusters:
        cmds.warning(f"{mesh_shape} Not detected skinCluster，跳过")
        continue
    skinCluster = skin_clusters[0]

    # inquiry influences
    influences = cmds.skinCluster(skinCluster, q=True, inf=True)
    J = len(influences)

    # Construct output dictionary
    data = {
        "mesh": transform,
        "skinCluster": skinCluster,
        "influences": influences,
        "joint_count": J
    }

    # sace：influences_xxx.json
    out_path = os.path.join(base_dir, f"influences_{transform}.json")
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f“Exported {transform} influences ({J} Joints) to {out_path}”)
