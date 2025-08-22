import maya.cmds as cmds
import json, os

# —— Configuration —— 
# Points to verts_rest.json in the same directory
base_dir = "/home/s5722875/Notes/Semester3/SkeletonFinished/maya_default_pose_output/weight"

# Meshes to be processed
meshes = cmds.ls(type='mesh', long=True)

for mesh in meshes:
    # get skinCluster
    hist = cmds.listHistory(mesh) or []
    sc = cmds.ls(hist, type='skinCluster')
    if not sc:
        cmds.warning(f"{mesh} dont have skinCluster，跳过")
        continue
    skinCluster = sc[0]

    # influences and J
    transform = cmds.listRelatives(mesh, parent=True, fullPath=False)[0]
    inf_j = json.load(open(os.path.join(base_dir, f"influences_{transform.replace('|','_')}.json")))
    influences = inf_j["influences"]
    J = len(influences)

    # vertex number V
    V = cmds.polyEvaluate(mesh, vertex=True)

    # Read weights
    # skinPercent Can be read in batches: returns a tiled list of [(joint1, w1), (joint2, w2), ...]
    w_data = []
    for vtx_id in range(V):
        vtx = f"{mesh}.vtx[{vtx_id}]"
        # value=True Pull all weights in joint order
        vals = cmds.skinPercent(skinCluster, vtx, q=True, value=True)
        # vals is J floating-point numbers returned in order of influence.
        w_data.append(vals)

    # JSON
    short_name = transform.split('|')[-1]  # Take the short name of transform
    out_path = os.path.join(base_dir, f"weights_{short_name}.json")
    with open(out_path, "w") as f:
        json.dump({
            "mesh": short_name,
            "V": V,
            "J": J,
            "weights": w_data
        }, f, indent=2)
    print(f“Weight matrix for {short_name} exported to {out_path}”)
