import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import process_pose as PP
from transforms3d.euler import euler2mat, mat2euler   # pip install transforms3d


def project_pose_to_2d(pose_globals_orig, K, M_cam_to_world):
    """
    Function to project 3D joint world positions onto 2D image pixel coordinates.

    Args:
        pose_globals_orig (dict):
            Keys are joint names, values are 4x4 numpy arrays (world transformation matrices).
        K (numpy.ndarray):
            3x3 camera intrinsic matrix.
        M_cam_to_world (dict):
            Camera extrinsic parameters dictionary, containing:
            - world_matrix: 4x4 list or array, the camera's world transformation matrix

    Returns:
        dict: {joint_name: (u, v)} pixel coordinates. Points behind the camera are filtered out.
    """
    # Extract and invert the camera world transformation matrix
    R_cam2world = M_cam_to_world[:3, :3]
    t_cam2world = M_cam_to_world[:3, 3]
    # Construct the transformation from world to camera: R = R_cam2world.T, t = -R @ t_cam2world
    R = R_cam2world.T
    t = -R @ t_cam2world

    joints_2d = {}
    for joint, M in pose_globals_orig.items():
        # Get the world coordinate position of the joint
        Xw = M[:3, 3]
        # World coordinate conversion to camera coordinates
        Xc = R @ Xw + t
        # Maya camera orientation is local -Z, so flip the Z axis so that positive values represent in front of the camera.
        Xc = np.array([Xc[0], Xc[1], -Xc[2]])
        Xc_x, Xc_y, Xc_z = Xc
        # Filter points behind the camera
        if Xc_z <= 0:
            continue
        # Perspective projection onto normalized image plane
        x_norm = Xc_x / Xc_z
        y_norm = Xc_y / Xc_z
        # Use the camera's internal reference to obtain pixel coordinates
        uvw = K @ np.array([x_norm, y_norm, 1.0])
        u, v = uvw[0], uvw[1]
        joints_2d[joint] = (u, v)

    return joints_2d

def normalize(v):
    """
    Normalize vector v.
    Args:
        v (np.ndarray): Input vector
    Returns:
        np.ndarray: Normalized vector
    """
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


def angle_between(u, v):
    """
    Calculate the angle between vectors u and v (in radians).
    Args:
        u (np.ndarray): Vector u
        v (np.ndarray): Vector v
    Returns:
        float: Angle, range [0, π]
    """
    dot = np.dot(u, v)
    norms = np.linalg.norm(u) * np.linalg.norm(v) + 1e-8
    return np.arccos(np.clip(dot / norms, -1.0, 1.0))

# ---------- Geometric Tools ----------
def sample_bezier_curve(ctrl, n=30):
    P0, P1, P2, P3 = ctrl
    ts = np.linspace(0.0, 1.0, n)
    return [((1-t)**3)*P0 + 3*(1-t)**2*t*P1 + 3*(1-t)*t**2*P2 + t**3*P3
            for t in ts]

def chain_curvature(pts):
    ang = 0.0
    for i in range(len(pts)-2):
        v1, v2 = pts[i+1]-pts[i], pts[i+2]-pts[i+1]
        v1 /= (np.linalg.norm(v1)+1e-8); v2 /= (np.linalg.norm(v2)+1e-8)
        ang += np.degrees(np.arccos(np.clip(v1.dot(v2), -1, 1)))
    return ang

def sample_cubic_bezier(ctrl, n=200):
    """
    ctrl: (P0, P1, P2, P3), each of which is (u, v)
    Returns curve sampling points of shape (n, 2)
    """
    P0, P1, P2, P3 = map(np.asarray, ctrl)
    ts = np.linspace(0.0, 1.0, n)[:, None]       # (n,1)
    omt = 1.0 - ts                               # one-minus-t
    curve = ((omt**3) * P0 +
             3 * (omt**2) * ts * P1 +
             3 *  omt      * ts**2 * P2 +
                 ts**3  * P3)
    return curve        # ndarray(n,2)
    
def _root_cam_depth(root_w, M_cam_to_world):
    """Returns Root in the camera coordinate system (x, y, z). Note that z = + forward."""
    Rcw = M_cam_to_world[:3, :3]         # world→cam R
    tcw = M_cam_to_world[:3, 3]          # world→cam T
    Xc = Rcw.T @ (root_w - tcw)          # world → cam
    Xc = np.array([Xc[0], Xc[1], -Xc[2]])# Maya camera -Z facing outward, flip
    return Xc                            # (x,y,z_cam)

def pixel_delta_to_cam(dU, dV, z, K):
    fx, fy = K[0,0], K[1,1]
    dx =  dU * z / fx
    dy =  dV * z / fy
    return np.array([dx, dy, 0.0])

def cam_to_world(delta_cam, M_cam_to_world):
    Rcw = M_cam_to_world[:3, :3]
    return Rcw @ delta_cam

def bezier_at_equal_height(ctrl, v_root, n=200):
    """Find the point on the Bézier curve that is at the same height as v_root and closest to it."""
    curve = sample_cubic_bezier(ctrl, n)     # shape (n,2)
    idx   = np.argmin(np.abs(curve[:, 1] - v_root))
    return curve[idx]                        # (u_tar, v_root≈)


def detect_curve_type_ctrl(ctrl, eps=1e-6):
    """
   Determine C/S based on only four control points.
    Return ‘C’ or ‘S’
    --------------------------------------
    eps  : Cross product absolute value < eps is considered collinear
    """
    P0, P1, P2, P3 = ctrl
    v03 = P3 - P0         # Baseline vector
    def side(P):
        # 2D cross product z component
        z = v03[0]*(P[1]-P0[1]) - v03[1]*(P[0]-P0[0])
        if abs(z) < eps:
            return 0 
        return 1 if z > 0 else -1
    s1, s2 = side(P1), side(P2)
    # If one of the points is basically on the line, revert to the previous broken line judgment.
    if s1 == 0 or s2 == 0:
        return 'C'
    return 'C' if s1 == s2 else 'S'

def add_world_yaw(root_xyz, dy):
    """Rotate the world by Y dy°; when dy==0, directly return the original list to avoid unnecessary copying."""
    if abs(dy) < 1e-6:
        return root_xyz
    Rx, Ry, Rz = np.radians(root_xyz)
    R_old = euler2mat(Rx, Ry, Rz, axes='sxyz')
    R_new = euler2mat(0, np.radians(dy), 0, axes='sxyz') @ R_old
    return np.degrees(mat2euler(R_new, axes='sxyz'))


def extract_loa_candidates(points_2d, align_max_deg=40, knee_straight_min_deg=135):
    """
    Based on 2D joint point coordinates, extract LoA candidates and retain joint names.

    Args:
        points_2d (dict): Joint name -> (u, v) pixel coordinates
        align_max_deg (float): Maximum angle threshold for alignment of extension lines (degrees)  
        knee_straight_min_deg (float): Minimum knee angle threshold for determining a “straight leg” (degrees)  

    Returns:  
        tuple:
            candidates (dict):
                ‘trunk_only’: [(joint, (u,v)), ...],
                Possible candidates for LoA, including ‘L_LEG’ or ‘R_LEG’
            selected_loa (list):
                Final selected LoA list, in the form (joint, (u,v))
    """
    # Angle to Radius
    theta_align_max = np.deg2rad(align_max_deg)
    theta_knee_straight = np.deg2rad(knee_straight_min_deg)

    # 1. Build a pure upper body LoA (including Root_M)
    trunk_seq = [
        'HeadEnd_M','Head_M','NeckPart1_M','Neck_M',
        'Chest_M','Spine1Part1_M','Spine1_M','RootPart1_M','Root_M'
    ]
    loa_trunk = []
    for j in trunk_seq:
        if j in points_2d:
            loa_trunk.append((j, tuple(map(float, points_2d[j]))))

    # 2. Calculate the main trend vector T (lower half of the main trunk)
    trend_seq = ['Spine1Part1_M','Spine1_M','RootPart1_M','Root_M']
    trend_pts = []
    for j in trend_seq:
        if j in points_2d:
            trend_pts.append(np.array(points_2d[j], dtype=float))
    V = []
    for i in range(len(trend_pts) - 1):
        d = trend_pts[i+1] - trend_pts[i]
        V.append(normalize(d))
    T = normalize(sum(V) / len(V)) if V else np.array([0.0, 0.0])

    # 3. Assess the alignment angle between the left and right legs and the T.
    leg_defs = {
        'L_LEG': ('Hip_L','Knee_L','Ankle_L'),
        'R_LEG': ('Hip_R','Knee_R','Ankle_R'),
    }
    align_angles = {}
    for leg, (hip, knee, _) in leg_defs.items():
        if hip in points_2d and knee in points_2d:
            v_hk = np.array(points_2d[knee]) - np.array(points_2d[hip])
            align_angles[leg] = angle_between(T, v_hk)

    # 4. Screen for legs that meet the alignment requirements
    passed = [leg for leg, ang in align_angles.items() if ang <= theta_align_max]
    selected_leg = None
    if len(passed) == 1:
        selected_leg = passed[0]
    elif len(passed) == 2:
        # If both conditions are met, take the smaller alignment angle.
        selected_leg = min(passed, key=lambda lg: align_angles[lg])

    # 5. Constructing a candidate dictionary
    candidates = {'trunk_only': loa_trunk}
    selected_loa = loa_trunk

    # 6. If there are leg candidates, generate LoA with legs.
    if selected_leg:
        hip, knee, ankle = leg_defs[selected_leg]
        p_hip = np.array(points_2d[hip], dtype=float)
        p_knee = np.array(points_2d[knee], dtype=float)
        p_ankle = np.array(points_2d[ankle], dtype=float)

        # Copy only the upper body
        loa_leg = loa_trunk.copy()
        # add hip, knee
        #loa_leg.append((hip, tuple(p_hip)))
        loa_leg.append((knee, tuple(p_knee)))

        # Assess knee alignment and possibly add ankle alignment
        theta_knee = angle_between(p_hip - p_knee, p_ankle - p_knee)
        if theta_knee >= theta_knee_straight:
            v_ka = p_ankle - p_knee
            if angle_between(T, v_ka) <= theta_align_max:
                loa_leg.append((ankle, tuple(p_ankle)))

        candidates[selected_leg] = loa_leg
        selected_loa = loa_leg

    return candidates, selected_loa

def fit_cubic_bezier(selected_loa):
    """
    Given a series of LoA points, fit a cubic Bézier curve:
      B(t) = (1−t)^3 P0 + 3(1−t)^2 t P1 + 3(1−t) t^2 P2 + t^3 P3
    where P0 and P3 are fixed as the first and last points, respectively, and find the optimal control points P1 and P2.

    Args:
        selected_loa: [(joint_name, (u,v)), …]

    Returns:
        control_points: (P0,P1,P2,P3)
        t_list: Parameters t_i corresponding to each observation point
    """
    # Extract coordinates
    pts = np.array([p for _, p in selected_loa], dtype=float)
    M = len(pts) - 1
    # 1) P0, P3
    P0 = pts[0]
    P3 = pts[-1]
    # 2) Parameterization of t_i: cumulative by string length
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(dists)])
    t_list = cum / cum[-1]  # Normalized to [0,1]

    # 3) Constructing the least squares: For i=1..M-1
    A = []
    Bx = []
    By = []
    for i in range(1, M):
        t = t_list[i]
        b0 = (1 - t)**3
        b1 = 3 * (1 - t)**2 * t
        b2 = 3 * (1 - t) * t**2
        b3 = t**3
        # We have: b0*P0 + b1*P1 + b2*P2 + b3*P3 = pts[i]
        # => b1*P1 + b2*P2 = pts[i] - b0*P0 - b3*P3
        A.append([b1, b2])
        rhs = pts[i] - (b0 * P0 + b3 * P3)
        Bx.append(rhs[0])
        By.append(rhs[1])
    A = np.array(A)           # shape (M-1, 2)
    Bx = np.array(Bx)         # shape (M-1,)
    By = np.array(By)

    # 4) Perform least squares for x and y dimensions separately.
    # [P1_x, P2_x] = lstsq(A, Bx)
    Px, *_ = np.linalg.lstsq(A, Bx, rcond=None)
    Py, *_ = np.linalg.lstsq(A, By, rcond=None)
    P1 = np.array([Px[0], Py[0]])
    P2 = np.array([Px[1], Py[1]])
    control_points = [P0, P1, P2, P3]
    return control_points, t_list

def _choose_push_side(p0, p3, root_uv):
    """
    Return +1 (right) / -1 (left) based on the left-right relationship between Root_M and the baseline (P0→P3).
    A cross product z component ≥0 is considered right.
    """
    v_line = p3 - p0
    z = v_line[0]*(root_uv[1]-p0[1]) - v_line[1]*(root_uv[0]-p0[0])
    return 1 if z >= 0 else -1


def get_ideal_bezier(control_points, pix_orig, loa_orig, pose_JR_orig,
                    sorted_entries, local_bind, parent_map, K, M_cam_to_world, 
                    push_policy='auto',CURVE_TH_C=25, CURVE_TH_S=55,push_thresh=45):
    """
    Automatically find a LoA Bézier curve that meets the curvature threshold.
    - Step-1: Directly detect the original curve
    - Step-2: Rotate the Root_M world Y-axis to find a more curved view
        - Step-3: If still insufficient, push P1/P2 along the normal
    - push_policy = ‘left’ | ‘right’ | ‘auto’
    Return: best_ctrl, best_pix, best_loa_pts
    """
        
    # ---------- Step-0: Put the “original” into the visualization placeholder first. ----------
    vis_pix, vis_loa = pix_orig, loa_orig
    best_ctrl        = control_points
    best_poseJR      = None
    # ---------- Step-1：original curve ----------
    pts0       = sample_bezier_curve(control_points)
    bend0      = chain_curvature(pts0)
    type0      = detect_curve_type_ctrl(control_points)
    thresh_now = CURVE_TH_C if type0 == 'C' else CURVE_TH_S
    print(f"[Init] {type0}-shape, bend={bend0:.1f}°, TH={thresh_now}°")

    if bend0 >= thresh_now:
        print("✅  The original curve has met the standard.")
        return control_points, vis_pix, vis_loa, pose_JR_orig   # No further steps required

    # ---------- Step-2：Rotate Root_M.Y ----------
    best_bend, best_ctrl, best_yaw = bend0, control_points, 0
    search_yaws = [0] + [d*s for d in range(5, 81, 5) for s in (1, -1)]
    for dy in search_yaws:
        pose_tmp            = copy.deepcopy(pose_JR_orig)
        pose_tmp['root_rot'] = add_world_yaw(pose_tmp.get('root_rot', [0,0,0]), dy)

        pose_glb   = PP.build_pose_globals(sorted_entries, local_bind,
                                           parent_map, pose_tmp, 'Root_M')
        pix_new    = project_pose_to_2d(pose_glb, K, M_cam_to_world)
        _, loa_new = extract_loa_candidates(pix_new, 35, 135)
        ctrl_new,_ = fit_cubic_bezier(loa_new)
        bend_new   = chain_curvature(sample_bezier_curve(ctrl_new))
        if bend_new > best_bend:
            best_bend, best_ctrl, best_yaw = bend_new, ctrl_new, dy
            best_pix, best_loa, best_poseJR = pix_new, loa_new, pose_tmp

    type_best   = detect_curve_type_ctrl(best_ctrl)
    thresh_best = CURVE_TH_C if type_best == 'C' else CURVE_TH_S

    if best_bend >= thresh_best:
        print(f"✅  Successful space remediation: {type_best}-shape, yaw={best_yaw:+}°, bend={best_bend:.1f}°")
        return best_ctrl, best_pix, best_loa, best_poseJR

    # ---------- Step-3：push P1/P2 ----------
    print("⚠️  Space remediation failed, start pushing P1/P2 …")
    P0, P1, P2, P3 = control_points
    v_line = P3 - P0
    n_unit = np.array([-v_line[1], v_line[0]]) / (np.linalg.norm(v_line)+1e-8)

    # ── Calculate the “push” direction ───────────────────────────────
    if push_policy == 'right':
        side =  1
    elif push_policy == 'left':
        side = -1
    else:   # 'auto'
        # Root_M 2D projection in the original pix_orig
        root_uv = pix_orig.get('Root_M', np.mean([P0, P3], axis=0))
        side = _choose_push_side(P0, P3, root_uv)

    n_unit *= side           # Unify by multiplying by +1 / -1 in the normal direction.
    best_s = None
    for s in np.linspace(10, 100, 10):
        ctrl_try = (P0, P1 + n_unit*s, P2 + n_unit*s, P3)
        bend_try = chain_curvature(sample_bezier_curve(ctrl_try))
        if bend_try > best_bend:
            best_bend, best_ctrl = bend_try, ctrl_try
        if best_bend >= push_thresh: 
            print(f"✅  Push Success: bend={best_bend:.1f}° (s={s:.0f}px)")
            break
    # ---------- Root Equal height shift (only when LoA is touched) ----------
    need_root_shift = any(j in ('Knee_L','Knee_R','Ankle_L','Ankle_R')
                          for j, _ in best_loa)

    if need_root_shift:
        best_poseJR = shift_root_to_bezier_equal_v(
            pose_JR_orig,
            best_ctrl, K, M_cam_to_world,
            sorted_entries, local_bind, parent_map
        )

    return best_ctrl, vis_pix, vis_loa, best_poseJR

def shift_root_to_bezier_equal_v(pose_JR, ctrl,
                                 K, M_cam_to_world,
                                 sorted_entries, local_bind, parent_map,
                                 n=200):
    """
    Only attach Root_M to the Bézier contour points in the u direction; v remains unchanged.
    """
    # ---- Current Root World Coordinates & Pixels ----
    pose_glb = PP.build_pose_globals(sorted_entries, local_bind,
                                     parent_map, pose_JR, 'Root_M')
    Xw_root = pose_glb['Root_M'][:3, 3]
    u_root, v_root = project_pose_to_2d({'Root_M': pose_glb['Root_M']},
                                        K, M_cam_to_world)['Root_M']

    # ---- Target pixel coordinates ----
    u_tar, _ = bezier_at_equal_height(ctrl, v_root, n)
    dU = u_tar - u_root                     # Only translate u
    if abs(dU) < 1e-4: 
        return pose_JR

    # ---- Pixel shift → World shift ----
    _, _, z_cam = _root_cam_depth(Xw_root, M_cam_to_world)
    delta_cam   = pixel_delta_to_cam(dU, 0.0, z_cam, K)
    delta_world = cam_to_world(delta_cam, M_cam_to_world)

    # ---- Back to pose_JR ----
    pose_new = copy.deepcopy(pose_JR)
    pose_new['root_pos'] = (np.asarray(pose_JR['root_pos']) + delta_world).tolist()
    return pose_new


def visualize_loa_pair(
        parent_map,
        # --------- before ----------
        pts2d_before, loa_before, ctrl_before,
        # --------- after  ----------
        pts2d_after , loa_after , ctrl_after ,
        *,
        num          = 100,
        figsize      = (7, 3),
        joint_color  = 'purple',
        skeleton_color='blue',
        loa_color    = 'red',
        bezier_color = 'orange',
        control_color= 'green',
        point_size   = 20,
        line_width   = 1.0,
        bezier_width = 2.0,
        annotate     = False,
        titles       = ('Before', 'After')
    ):
    """
    Visualize the two sets of data, “before optimization” and “after optimization,” side by side.
    ------------------------------------------------------------------------
    pts2d_* : dict{joint:(u,v)}
    loa_*   : [(joint,(u,v)), ...]  or None
    ctrl_*  : (P0,P1,P2,P3)         or None
    titles  : Titles of the two subplots
    """
    def _draw_single(ax, pts2d, loa, ctrl, title):
        # Joint + Skeleton
        xs, ys = zip(*pts2d.values()) if pts2d else ([], [])
        ax.scatter(xs, ys, s=point_size, color=joint_color)
        for j, (u, v) in pts2d.items():
            p = parent_map.get(j)
            if p in pts2d:
                pu, pv = pts2d[p]
                ax.plot([u, pu], [v, pv], color=skeleton_color,
                        linewidth=line_width)
            if annotate:
                ax.text(u, v, j, fontsize=6, color=joint_color)

        # LoA candidate
        if loa:
            ax.plot([p[1][0] for p in loa],
                    [p[1][1] for p in loa],
                    color=loa_color, linewidth=line_width, label='LoA')
            if annotate:
                for j, (u, v) in loa:
                    ax.text(u, v, j, fontsize=6, color=loa_color)

        # Bézier Curves & Control Points
        if ctrl is not None:
            P0, P1, P2, P3 = ctrl
            ts = np.linspace(0, 1, num)
            curve = np.array([(1-t)**3 * P0 + 3*(1-t)**2*t*P1 +
                               3*(1-t)*t**2*P2 + t**3*P3 for t in ts])
            ax.plot(curve[:, 0], curve[:, 1], color=bezier_color,
                    linewidth=bezier_width, label='Bezier')
            cps = np.vstack([P0, P1, P2, P3])
            ax.plot(cps[:, 0], cps[:, 1], '--o', color=control_color,
                    label='Ctrl')

        ax.set_title(title)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.legend(loc='lower right', fontsize=8)

    # ----------------------- Draw two subgraphs -----------------------
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _draw_single(axes[0], pts2d_before, loa_before, ctrl_before, titles[0])
    _draw_single(axes[1], pts2d_after , loa_after , ctrl_after , titles[1])
    plt.tight_layout()
    plt.show()
