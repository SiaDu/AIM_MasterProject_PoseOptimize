import maya.cmds as cmds
import json
import os

# —— 1. 获取所有目标 Mesh —— 
# 如果你想处理场景里所有 mesh，可以用：
meshes = cmds.ls(type='mesh', long=True)
# 如果你只想处理当前选中的，可以改成：
# meshes = cmds.ls(selection=True, dag=True, shapes=True, type='mesh', long=True)

if not meshes:
    raise RuntimeError("在场景中没有检测到任何 Mesh（或请先选中要处理的 Mesh）")

# —— 2. 准备导出目录 —— 
base_dir = "/home/s5722875/Notes/Semester3/SkeletonFinished/maya_default_pose_output/weight"
os.makedirs(base_dir, exist_ok=True)

for mesh_shape in meshes:
    # 从 shape 名称推 transform 名称（可选，用于文件命名）
    transform = cmds.listRelatives(mesh_shape, parent=True, fullPath=False)[0]

    # 找到连接到这个 Mesh 的 skinCluster
    hist = cmds.listHistory(mesh_shape) or []
    skin_clusters = cmds.ls(hist, type='skinCluster')
    if not skin_clusters:
        cmds.warning(f"{mesh_shape} 没有检测到 skinCluster，跳过")
        continue
    skinCluster = skin_clusters[0]

    # 查询 influences
    influences = cmds.skinCluster(skinCluster, q=True, inf=True)
    J = len(influences)

    # 构造输出字典
    data = {
        "mesh": transform,
        "skinCluster": skinCluster,
        "influences": influences,
        "joint_count": J
    }

    # 存文件：influences_xxx.json
    out_path = os.path.join(base_dir, f"influences_{transform}.json")
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"已导出 {transform} 的 influences（{J} 个 Joint）到 {out_path}")
