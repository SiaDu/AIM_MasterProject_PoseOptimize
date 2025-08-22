import maya.cmds as cmds

"""Just use import, set the pose for the adv rigged character, and then use it 
in the current frame of the animation."""

def rebuild_clean_pose_and_weights(
    source_geo_grp='geo|geo_grp|girl_geo_grp',
    source_root_joint='Root_M',
    target_geo_grp='girl_geo_grp1',
    target_root_joint='Root_M1'
):
    """
    One-click process:
    1. Duplicate geometry and joint to world space;
    2. Bind mesh to new joint;
    3. Copy skin weights from original;
    4. Delete original groups (e.g. 'geo', 'Group');
    5. Rename new objects to final standard names.
    """
    # === Step 1: Duplicate geo and skeleton ===
    new_geo_grp = cmds.duplicate(source_geo_grp, rr=True)[0]
    cmds.parent(new_geo_grp, world=True)
    cmds.reorder(new_geo_grp, relative=-4)
    new_geo_grp = cmds.rename(new_geo_grp, target_geo_grp)

    new_root = cmds.duplicate(source_root_joint, rr=True)[0]
    cmds.parent(new_root, world=True)
    cmds.reorder(new_root, relative=-3)
    new_root = cmds.rename(new_root, target_root_joint)

    print(f"✅ Duplicated: {new_geo_grp}, {new_root}")

    # === Step 2: Bind skin (empty weights) ===
    mesh_shapes = cmds.listRelatives(new_geo_grp, allDescendents=True, type='mesh', fullPath=True)
    mesh_transforms = list({cmds.listRelatives(s, parent=True, fullPath=True)[0] for s in mesh_shapes})

    joint_list = cmds.listRelatives(new_root, allDescendents=True, type='joint', fullPath=True) or []
    joint_list.insert(0, new_root)

    for mesh in mesh_transforms:
        try:
            cmds.skinCluster(
                joint_list,
                mesh,
                toSelectedBones=False,
                bindMethod=0,
                skinMethod=0,
                normalizeWeights=1,
                weightDistribution=0,
                maximumInfluences=3,
                dropoffRate=4.0
            )
            print(f"✅ Bound: {mesh}")
        except Exception as e:
            print(f"❌ Binding failed: {mesh} - {e}")

    # === Step 3: Copy skin weights by name ===
    source_shapes = cmds.listRelatives(source_geo_grp, allDescendents=True, type='mesh', fullPath=True)
    source_transforms = [cmds.listRelatives(s, parent=True, fullPath=True)[0] for s in source_shapes]

    target_shapes = cmds.listRelatives(new_geo_grp, allDescendents=True, type='mesh', fullPath=True)
    target_transforms = [cmds.listRelatives(s, parent=True, fullPath=True)[0] for s in target_shapes]

    source_dict = {s.split('|')[-1]: s for s in source_transforms}
    target_dict = {t.split('|')[-1]: t for t in target_transforms}

    print("\n⏳ Copying skin weights...")
    for name in source_dict:
        if name in target_dict:
            src = source_dict[name]
            tgt = target_dict[name]
            skin = cmds.ls(cmds.listHistory(src), type='skinCluster')
            if not skin:
                print(f"⚠️ Skipped (no skinCluster): {name}")
                continue
            try:
                cmds.select(src, r=True)
                cmds.select(tgt, add=True)
                cmds.copySkinWeights(
                    noMirror=True,
                    surfaceAssociation='closestPoint',
                    influenceAssociation=['closestJoint', 'oneToOne']
                )
                print(f"✅ Weights copied: {name}")
            except Exception as e:
                print(f"❌ Failed to copy weights: {name} - {e}")
        else:
            print(f"❌ No matching target mesh: {name}")

    # === Step 4: Delete original groups ===
    for grp in ['geo', 'Group']:
        if cmds.objExists(grp):
            try:
                cmds.delete(grp)
                print(f"🗑️ Deleted group: {grp}")
            except Exception as e:
                print(f"⚠️ Failed to delete group {grp}: {e}")

    # === Step 5: Rename final outputs ===
    final_geo_name = 'girl_geo_grp'
    final_joint_name = 'Root_M'
    if cmds.objExists(final_geo_name):
        cmds.delete(final_geo_name)
    if cmds.objExists(final_joint_name):
        cmds.delete(final_joint_name)
    new_geo_grp = cmds.rename(new_geo_grp, final_geo_name)
    new_root = cmds.rename(new_root, final_joint_name)

    print("\n🎉 All steps completed!")
    print(f"✅ Final geo group: {new_geo_grp}")
    print(f"✅ Final root joint: {new_root}")

rebuild_clean_pose_and_weights(
    source_geo_grp='geo',
    source_root_joint='Root_M',
    target_geo_grp='girl_geo_grp1',
    target_root_joint='Root_M1'
)