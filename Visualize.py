import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

def plot_skeleton(All_joint_global_matrix, parent_map, img_name = None,
                  elev=90, azim=-90, figsize=(8,8), point_size=20, line_width=1):
    """
    Draw the skeleton on a new 3D axis:
      - joint_positions Extract the world translation of each joint from All_joint_global_matrix.
      - Connect the skeleton tree defined by parent_map.
      - You can customize the viewing angle elev and azim.

    Parameters:
      All_joint_global_matrix: dict[str, np.ndarray(4x4)]  World matrix for each joint
      parent_map:              dict[str, str|None]         parent node short name (the value for the root is None)
    """
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')

    # Extract the world positions of all relevant nodes.
    joint_positions = {j: All_joint_global_matrix[j][:3, 3] for j in All_joint_global_matrix}

    # Draw all joints
    for j, pos in joint_positions.items():
        ax.scatter(pos[0], pos[1], pos[2], s=point_size)
        ax.text(pos[0], pos[1], pos[2], j, size=6)

    # Drawing the connection between father and son
    for j, pos in joint_positions.items():
        parent = parent_map.get(j)
        if parent:
            ppos = joint_positions[parent]
            ax.plot([pos[0], ppos[0]],
                    [pos[1], ppos[1]],
                    [pos[2], ppos[2]],
                    color='blue', linewidth=line_width)

    # Center & Scale proportionally
    all_pts = np.array(list(joint_positions.values()))
    center  = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    radius  = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    # Set perspective
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(img_name)
    plt.tight_layout()
    plt.show()

def plot_skeleton_and_skin(
        joint_global_matrix: dict,
        parent_map: dict,
        V_concat: np.ndarray | None = None,      # (N,3) Deformed vertex
        skin_point_size: float = 0.4,            # Vertex scatter size
        max_skin_points: int = 100_000, 
        elev: int = 90, azim: int = -90,
        figsize=(8, 8),
        joint_point_size: int = 20,
        joint_line_width: int = 1,
        show_joint_labels: bool = False
    ):
    """
    Simultaneously visualize skeleton & skin point clouds in the same image
    ------------------------------------------------------------
    - joint_global_matrix : {joint: 4×4 worldMatrix}
    - parent_map          : {joint: parent_joint or None}
    - V_concat            : (N,3)  All vertices of the character (can be obtained using skin_character)
                             If None, only the skeleton is drawn
    """
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')

    # ---------- 1. plot skeleton ----------
    joint_pos = {j: joint_global_matrix[j][:3, 3] for j in joint_global_matrix}

    for j, pos in joint_pos.items():
        ax.scatter(*pos, s=joint_point_size, c='red')
        if show_joint_labels:
            ax.text(*pos, j, size=6)

    for j, pos in joint_pos.items():
        p = parent_map.get(j)
        if p:
            ppos = joint_pos[p]
            ax.plot([pos[0], ppos[0]],
                    [pos[1], ppos[1]],
                    [pos[2], ppos[2]],
                    c='blue', lw=joint_line_width)

    # ---------- 2. plot skin point clouds ----------
    if V_concat is not None and len(V_concat):
        if len(V_concat) > max_skin_points:
            idx = random.sample(range(len(V_concat)), max_skin_points)
            V_sub = V_concat[idx]
        else:
            V_sub = V_concat

        ax.scatter(V_sub[:, 0], V_sub[:, 1], V_sub[:, 2],
                   s=skin_point_size, c='gray', alpha=0.7)

    # ---------- 3. Centered proportionally ----------
    # Merge bone & skin coordinates to find bounding box
    all_pts = np.vstack([*joint_pos.values(),
                         V_concat if V_concat is not None else []])

    center  = (all_pts.max(0) + all_pts.min(0)) / 2
    radius  = (all_pts.max(0) - all_pts.min(0)).max() / 2

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Skeleton + Skinned Mesh')
    plt.tight_layout()
    plt.show()
    return fig, ax