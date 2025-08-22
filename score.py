import numpy as np

def camera_facing_score(pose_globals: dict,
                        cam_global: np.ndarray,
                        joint_name: str = "Head_M"):
    """
    Calculate the score of the specified joint facing the camera, range [0,1]:
      1 indicates fully facing (angle 0°),
      0 indicates facing away or 90° or more.
    
    Parameters:
      pose_globals: Global transformation dictionaries for each joint, represented 
      as 4x4 np.ndarray matrices
      cam_global: Camera's world matrix (4x4 np.ndarray)
      joint_name: Name of the joint to be scored, default is “Head_M”
    
    Returns:
      score: Orientation score, floating-point number
    """
    # 1) Remove head position & Z-axis orientation
    head_T = pose_globals[joint_name] #"Head_M"
    head_pos = head_T[:3, 3] # joint 的tx,ty,tz
    head_z  = head_T[:3, 1] # Direction of the joint Z-axis in the world coordinate system
    
    # 2) Remove the camera position
    cam_pos = cam_global[:3, 3] # cam tx,ty,tz
    
    # 3) Calculate vector & normalize  
    to_cam = cam_pos - head_pos # [dx, dy, dz](Uniform length and direction)
    v1 = to_cam / np.linalg.norm(to_cam) 
    v2 = head_z  / np.linalg.norm(head_z)
    
    # 4) Dot product & inverse cosine to obtain angle (radians), cosine value as a fraction, clamped to [0,1]
    cosang = np.dot(v1, v2)
    angle_rad = np.arccos(cosang)
    angle_deg = np.degrees(angle_rad)
    score  = max(0.0, min(1.0, cosang))
    
    return score
    
#score = camera_facing_score(pose_globals_orig,cam_global,joint_name="Head_M")

def line_of_action_score(loa_pts, bezier_ctrl,
                         n_sample    = 200,
                         lambda_fit  = 0.3,
                         agg_roothip = True):
    """
    loa_pts      : [(joint,(u,v)), ...] —— 2-D LoA point sequence
    bezier_ctrl  : (P0,P1,P2,P3)
    n_sample     : Bézier sampling density
    lambda_fit   : Coefficient mapping distance to score; higher values impose heavier penalties
    agg_roothip  : True → Aggregate Hip_L / Hip_R to the midpoint of Root_M
    ----------------------------------------------------
    Returns S_fit ∈ (0,1], the closer to 1 the better the fit
    """
    def sample_bezier(ctrl, n=100):
        """ctrl = (P0,P1,P2,P3)，return n×2 array"""
        P0, P1, P2, P3 = [np.asarray(p, float) for p in ctrl]
        t  = np.linspace(0.0, 1.0, n)[:, None]      # (n,1)
        one_minus_t = 1 - t
        curve = (one_minus_t**3) * P0 + \
                3 * (one_minus_t**2) * t * P1 + \
                3 *  one_minus_t     * t**2 * P2 + \
                t**3 * P3
        return curve                     # (n,2)
    # -------- 1. Prepare LoA point (optional hip aggregation) --------
    pts = []
    root_xy = None
    for j, xy in loa_pts:
        p = np.asarray(xy, float)
        if j == 'Root_M':
            root_xy = p
            pts.append(p)
        elif agg_roothip and j in ('Hip_L', 'Hip_R') and root_xy is not None:
            pts.append((p + root_xy) / 2.0)     # Root–Hip mid point
        else:
            pts.append(p)
    pts = np.vstack(pts)                        # (m,2)

    # -------- 2. Bézier curve sampling --------
    curve = sample_bezier(bezier_ctrl, n_sample)   # (n,2)

    # -------- 3. Calculate the closest distance for each point --------
    # Find the nearest point on the sampling curve for each LoA point.
    dists2 = []
    for p in pts:
        d2 = np.sum((curve - p)**2, axis=1)        # For all sampling points
        dists2.append(d2.min())
    E_fit = float(np.mean(dists2))                 # mean square error

    # -------- 4. Mapped to a score between 0~1 --------
    S_fit = 1.0 / (1.0 + lambda_fit * E_fit)
    return S_fit, E_fit, np.sqrt(dists2)           # Also return error details

"===================================Silhouette Part======================================="

"-------------------------A) Segment definition + gadgets----------------------------"
# —— Segment by naming according to your skeleton (the influence set is used to select 
# vertices belonging to that segment based on skin weights).——
SEG_DEFS = {
    'torso'     : ('RootPart1_M','Chest_M', {'RootPart1_M','Spine1_M','Spine1Part1_M','Chest_M'}),
    'thigh_L'   : ('Hip_L','Knee_L',        {'Hip_L','HipPart1_L','Knee_L'}),
    'calf_L'    : ('Knee_L','Ankle_L',      {'Knee_L','Ankle_L'}),
    'thigh_R'   : ('Hip_R','Knee_R',        {'Hip_R','HipPart1_R','Knee_R'}),
    'calf_R'    : ('Knee_R','Ankle_R',      {'Knee_R','Ankle_R'}),
    'upperarm_L': ('Shoulder_L','Elbow_L',  {'Shoulder_L','ShoulderPart1_L','Elbow_L'}),
    'forearm_L' : ('Elbow_L','Wrist_L',     {'Elbow_L','ElbowPart1_L','Wrist_L'}),
    'upperarm_R': ('Shoulder_R','Elbow_R',  {'Shoulder_R','ShoulderPart1_R','Elbow_R'}),
    'forearm_R' : ('Elbow_R','Wrist_R',     {'Elbow_R','ElbowPart1_R','Wrist_R'}),
}

def _point_segment_dist_and_t(P, A, B):
    """
    Find the closest point on line segment AB and return the parameter t (the position 
    of the projection point on the line segment, 0 at A, 1 at B).
    """
    v = B - A 
    vv = float(np.dot(v, v)) + 1e-12
    t = float(np.dot(P - A, v) / vv)
    t = max(0.0, min(1.0, t))
    proj = A + t * v
    return float(np.linalg.norm(P - proj)), t

"--B) T-pose：Estimate the radius of each segment of the world using per_vertex_weights + V_concat. ---"
def precompute_segment_r_world_from_weights(Tpose_globals, V_concat, per_vertex_weights,
    *, quantile=0.80, safety=1.10, weight_thr=0.30, use_cone=True, min_samples=50):
    """
    Function: Use per_vertex_weights + V_concat in T-pose world coordinates to estimate the 
        “radius” (thickness) for each segment.  
    V_concat: All vertex positions in T-pose (Nx3).  
    per_vertex_weights: The “joint → weight” dictionary for each vertex.
    * The parameters on the left can be passed by position (positional) or by name (keyword).
    * The parameters on the right can only be passed by keyword, not by position.
    quantile (quantile, statistical concept): Use the 80% quantile as the radius, which 
        is more robust than the maximum value (not affected by outliers).
    safety: Safety factor, multiply the estimated radius by a small number >1 to leave a margin.
    weight_thr: Threshold for the “related weight sum” of vertices belonging to a segment.  
    use_cone=True: Each segment uses a “conical radius” (radius r0 near the starting point, 
        radius r1 near the endpoint), otherwise a constant radius is used.  
    min_samples: If too few points are selected for a segment, it reverts to using “all vertices”
        for statistics to avoid insufficient samples.
    """
    # Take the translation column (world coordinates) from the 4x4 transformation of the 
    # T-pose joints and use it as the two ends of the “line segment.”
    seg_lines = {}
    for seg, (j0, j1, _) in SEG_DEFS.items():
        if j0 in Tpose_globals and j1 in Tpose_globals:
            A = Tpose_globals[j0][:3, 3]
            B = Tpose_globals[j1][:3, 3]
            seg_lines[seg] = (A, B)
    # One-to-one correspondence: The position and weight of the i-th vertex must be aligned.
    N = len(V_concat)
    assert N == len(per_vertex_weights), "V_concat and per_vertex_weights have inconsistent lengths!"

    radii = {} # Final return: Each segment → Radius parameter
    all_idx = np.arange(N) # Alternative: Fall back on a backup plan when there aren't enough points to choose from.

    for seg, (A, B) in seg_lines.items():
        jset = SEG_DEFS[seg][2] #The “relevant joint set” in the current segment (e.g., thigh: Hip/Thigh/Knee).

        # —— Select candidate vertices for the segment (based on weight) ——
        idx = []
        for i, wdict in enumerate(per_vertex_weights):
            if not wdict: 
                continue
            sw = sum(float(wdict.get(j, 0.0)) for j in jset)  # Sum of relevant joint weights
            if sw >= weight_thr:
                idx.append(i); continue
            # Maximum weight allocation as a safety net
            jmax = max(wdict.items(), key=lambda kv: kv[1])[0]
            if jmax in jset:
                idx.append(i)

        cand = np.array(idx, dtype=int) if idx else all_idx
        """
        Calculate the distance distribution from these points to the line segment. 
        For candidate vertices, calculate their closest distance d to the segment and the 
        landing point position t ([0,1]).
        """
        P = V_concat[cand]
        dists, ts = [], []
        for p in P:
            d, t = _point_segment_dist_and_t(p, A, B)
            dists.append(d); ts.append(t)
        dists = np.asarray(dists); ts = np.asarray(ts)

        if len(dists) < min_samples:
            # Bottom line: All members
            P = V_concat
            dists, ts = [], []
            for p in P:
                d, t = _point_segment_dist_and_t(p, A, B)
                dists.append(d); ts.append(t)
            dists = np.asarray(dists); ts = np.asarray(ts)

        eps = 1e-4
        if use_cone:
            left  = dists[ts < 0.5] if (ts < 0.5).any() else dists
            right = dists[ts >= 0.5] if (ts >= 0.5).any() else dists
            r0 = float(np.quantile(left,  quantile) * safety)
            r1 = float(np.quantile(right, quantile) * safety)
            radii[seg] = {'type':'cone','r0':max(r0,eps),'r1':max(r1,eps)}
        else:
            r = float(np.quantile(dists, quantile) * safety)
            radii[seg] = {'type':'capsule','r':max(r,eps)}
    return radii

"------------------C) Runtime: World radius → Pixel radius; Build 2D capsule----------------------------"
def _world_to_cam(xw, M_cam_to_world):
    Rcw = M_cam_to_world[:3, :3]; tcw = M_cam_to_world[:3, 3]
    Xc = Rcw.T @ (xw - tcw)
    return np.array([Xc[0], Xc[1], -Xc[2]])  # z>0 Indicates in front of the camera (in conjunction with project's -Z camera)

def _world_radius_to_px(r_world, z_cam, K):
    fx = float(K[0,0])
    return fx * float(r_world) / max(z_cam, 1e-6)

def capsule_r_px_runtime(pose_globals_now, K, M_cam_to_world, radii_world):
    """At runtime, take the endpoints A/B of the line segment from the global matrix of the current 
    pose; calculate their depth zA and zB relative to the camera (z > 0 means they are in front)."""
    r_px = {}
    for seg, (j0,j1,_) in SEG_DEFS.items():
        if seg not in radii_world: 
            continue
        if j0 not in pose_globals_now or j1 not in pose_globals_now:
            continue
        A = pose_globals_now[j0][:3,3]
        B = pose_globals_now[j1][:3,3]
        zA = _world_to_cam(A, M_cam_to_world)[2]
        zB = _world_to_cam(B, M_cam_to_world)[2]
        if zA <= 0 or zB <= 0:
            continue
        # Convert the world radius to pixel radius. Convert both ends of the cone and take the
        #  larger value to be on the safe side. Use the middle depth to approximate the constant radius.
        rw = radii_world[seg]
        if rw['type'] == 'cone':
            rA = _world_radius_to_px(rw['r0'], zA, K)
            rB = _world_radius_to_px(rw['r1'], zB, K)
            r_px[seg] = float(max(rA, rB))   # 安全取大
        else:
            r_mid = _world_radius_to_px(rw['r'], 0.5*(zA+zB), K)
            r_px[seg] = float(r_mid)
    return r_px

def build_capsules_px(points_2d, r_px_by_seg):
    """Connect the 2D key points into “line segments” and add pixel radii to obtain 2D capsules 
    (line segments + radii)."""
    C = {}
    def _uv(j): 
        return np.array(points_2d[j], dtype=float) if j in points_2d else None
    for seg, (j0,j1,_) in SEG_DEFS.items():
        if seg not in r_px_by_seg: 
            continue
        p0 = _uv(j0); p1 = _uv(j1)
        if p0 is None or p1 is None:
            continue
        C[seg] = dict(p0=p0, p1=p1, r=float(r_px_by_seg[seg]))
    # Add an extra point at the elbow (can also use the forearm capsule).
    if 'Elbow_L' in points_2d:
        C['elbow_L_pt'] = dict(pt=np.array(points_2d['Elbow_L'], float), r=r_px_by_seg.get('forearm_L', 6.0))
    if 'Elbow_R' in points_2d:
        C['elbow_R_pt'] = dict(pt=np.array(points_2d['Elbow_R'], float), r=r_px_by_seg.get('forearm_R', 6.0))
    return C

"-------------------------D) 2D capsule geometry & silhouette rating----------------------------"
# —— 2D geometry —— 
def _clamp01(x): return max(0.0, min(1.0, x))

def _seg_seg_min_dist(p, q, r, s):
    "Function: Minimum distance between two 2D line segments."
    u = q - p; v = s - r; w0 = p - r
    a = np.dot(u,u) + 1e-12
    b = np.dot(u,v)
    c = np.dot(v,v) + 1e-12
    d = np.dot(u,w0)
    e = np.dot(v,w0)
    D = a*c - b*b
    sc = _clamp01((b*e - c*d)/D) if D>1e-12 else 0.0
    tc = _clamp01((a*e - b*d)/D) if D>1e-12 else 0.0
    cp = p + sc*u; cq = r + tc*v
    return np.linalg.norm(cp - cq)

def capsule_minsep(capA, capB):
    "Click on the capsule's “minimum boundary spacing.” Similarly: distance − radius."
    d = _seg_seg_min_dist(capA['p0'], capA['p1'], capB['p0'], capB['p1'])
    return d - (capA['r'] + capB['r'])

def point_capsule_minsep(pt, cap):
    p, q = cap['p0'], cap['p1']
    v = q - p; t = np.dot(pt - p, v) / (np.dot(v,v) + 1e-12)
    t = _clamp01(t); closest = p + t*v
    d = np.linalg.norm(pt - closest)
    return d - cap['r']

# —— Unified comparison of any two types (line vs. line, point vs. line, point vs. point)
def _minsep_any(itemA, itemB):
    # item might be {'p0','p1','r'} 或 {'pt','r'}
    is_ptA = ('pt' in itemA); is_ptB = ('pt' in itemB)
    if not is_ptA and not is_ptB:
        return capsule_minsep(itemA, itemB)
    if is_ptA and not is_ptB:
        return point_capsule_minsep(itemA['pt'], itemB)
    if not is_ptA and is_ptB:
        return point_capsule_minsep(itemB['pt'], itemA)
    # pt vs pt
    d = np.linalg.norm(itemA['pt'] - itemB['pt'])
    return d - (itemA['r'] + itemB['r'])

# —— Take the minimum distance between the two sets of names + record who is the smallest
def _minsep_groups(C, namesA, namesB):
    best = None; arg = None
    for a in namesA:
        if a not in C: 
            continue
        for b in namesB:
            if b not in C: 
                continue
            s = _minsep_any(C[a], C[b])
            if (best is None) or (s < best):
                best, arg = s, (a, b)
    return best, arg  # best might be None

# Organize the “names to be compared.” List the capsule names involved in each 
# leg/arm (including upper arm, forearm, elbow point, etc.).
def _names_leg(side):  # 'L' or 'R'
    return [f'thigh_{side}', f'calf_{side}']

def _names_arm(side):  # 'L' or 'R'
    # Consider both line capsules and elbow points (elbow_*_pt has already been added to build_capsules_px).
    return [f'upperarm_{side}', f'forearm_{side}', f'elbow_{side}_pt']


# —— Select LoA leg side (your LoA output is sufficient)——
def loa_leg_side(selected_loa):
    """Check whether the LoA includes the left/right leg joints; if it only includes one side, 
    that side is the “main leg,” and the other side should be separated first (to reduce damage to the LoA)."""
    names = [j for j,_ in selected_loa] if selected_loa else []
    if any(n in names for n in ('Knee_L','Ankle_L')) and not any(n in names for n in ('Knee_R','Ankle_R')):
        return 'L'
    if any(n in names for n in ('Knee_R','Ankle_R')) and not any(n in names for n in ('Knee_L','Ankle_L')):
        return 'R'
    return None  # trunk-only or both leg

# —— score —— 
def gap_reward(minsep, target_px=12.0, overlap_penalty=2.0):
    """The goal is to achieve “minimum spacing ≥ target_px”.
    If the spacing is negative (overlap), a negative score is given, with a maximum deduction of 
        -overlap_penalty.
    If the spacing is positive, the score increases proportionally, with a maximum of 1. (Convert 
        the geometric value to a reward between 0 and 1)."""
    if minsep < 0:
        return max(-overlap_penalty, minsep/(-target_px))
    return min(1.0, minsep/target_px)

def silhouette_terms_from_capsules(C):
    torso = C.get('torso', None)
    if torso is None:
        return dict(legL=None, legR=None, elbowL=None, elbowR=None)

    def leg_side(side):
        thigh = C.get(f'thigh_{side}'); calf = C.get(f'calf_{side}')
        vals = []
        if thigh is not None: vals.append(capsule_minsep(thigh, torso))
        if calf  is not None: vals.append(capsule_minsep(calf,  torso))
        return min(vals) if vals else None

    legL = leg_side('L'); legR = leg_side('R')

    def elbow_side(side):
        fore = C.get(f'forearm_{side}')
        if fore is not None:
            return capsule_minsep(fore, torso)
        pt = C.get(f'elbow_{side}_pt', None)
        if pt is None: return None
        return point_capsule_minsep(pt['pt'], torso)

    elbowL = elbow_side('L'); elbowR = elbow_side('R')
    return dict(legL=legL, legR=legR, elbowL=elbowL, elbowR=elbowR)

def silhouette_terms_from_capsules_v2(C):
    terms  = {}
    argmin = {}

    torso_names = ['torso'] if ('torso' in C) else []

    # —— Trunk vs. legs (left and right) ——
    for s in ('L', 'R'):
        names_leg = _names_leg(s)
        val, arg = _minsep_groups(C, torso_names, names_leg)
        terms[f'torso_leg_{s}'] = val
        argmin[f'torso_leg_{s}'] = arg

    # —— Trunk vs. arms (left and right) ——
    for s in ('L', 'R'):
        names_arm = _names_arm(s)
        val, arg = _minsep_groups(C, torso_names, names_arm)
        terms[f'torso_arm_{s}'] = val
        argmin[f'torso_arm_{s}'] = arg

    # —— Leg vs. leg (L–R, take the minimum on the straddle side) ——
    val, arg = _minsep_groups(C, _names_leg('L'), _names_leg('R'))
    terms['leg_leg'] = val
    argmin['leg_leg'] = arg

    # —— Arm vs. Arm (L–R) ——
    val, arg = _minsep_groups(C, _names_arm('L'), _names_arm('R'))
    terms['arm_arm'] = val
    argmin['arm_arm'] = arg

    # —— Arm vs. leg (more common on the opposite side: left arm vs. right leg, right arm vs. left leg)——
    v1, a1 = _minsep_groups(C, _names_arm('L'), _names_leg('R'))
    v2, a2 = _minsep_groups(C, _names_arm('R'), _names_leg('L'))
    # Take the smaller of the two cross-side matches.
    if (v1 is None) and (v2 is None):
        terms['arm_leg'] = None; argmin['arm_leg'] = None
    elif (v2 is None) or (v1 is not None and v1 <= v2):
        terms['arm_leg'] = v1;   argmin['arm_leg'] = a1
    else:
        terms['arm_leg'] = v2;   argmin['arm_leg'] = a2

    return terms, argmin
    
def silhouette_score(points_2d, r_px_by_seg, keep_ctrl=None, selected_loa=None, 
                     target_gap_px=10, w_leg_torso=0.0, w_elbow_torso=0.0, w_keep=0.0,
                     w_arm_torso=0.0, w_leg_leg=0.0, w_arm_arm=0.0, w_arm_leg=0.0,
                     tgt_leg_torso=None, tgt_arm_torso=None, tgt_leg_leg=None, tgt_arm_arm=None, 
                     tgt_arm_leg=None):
    """
    target_gap_px: Global default “desired minimum gap width” (pixels). Larger for close-ups (12–18) 
        and smaller for long shots (6–10).
    w_leg_torso: Weight for the “leg vs. torso” factor (primary factor, default 0.7). Scores are applied 
        only to the selected movable leg (priority is given to the leg not aligned with the LoA).  
    w_elbow_torso: Weight for the “elbow vs. torso” factor; keep it at 0.
    w_keep: Weight for “maintain LoA direction” (use your keep_ctrl as a reference). The higher the value, 
        the less likely the yellow line will be distorted.  
    w_arm_torso: Weight for “arm (upper arm/forearm/elbow point) vs. torso.” Add a bit if the arm is always 
        close to the body (default 0.15, recommended 0.1–0.3).
    w_leg_leg: Weight for “leg vs. leg (left and right legs relative to each other).” Add when the legs 
        overlap significantly (default 0.15).  
    w_arm_arm: Weight for “arm vs. arm (left and right arms relative to each other).” Add when you don't 
        want the arms to stick together except when hugging (default 0.15).
    w_arm_leg: Weight for “arm vs. leg (cross-side priority, e.g., left arm-right leg).” Add when common 
        cross-obscuring occurs (default 0.15).  
    tgt_leg_torso / tgt_arm_torso / tgt_leg_leg / tgt_arm_arm / tgt_arm_leg: Target seam width for each 
        category; defaults to specified ratio if not filled:
    leg_torso: = target_gap_px
    arm_torso: ≈ 0.7 * target_gap_px
    leg_leg: ≈ 0.8 * target_gap_px
    arm_arm: ≈ 0.7 * target_gap_px
    arm_leg: ≈ 0.8 * target_gap_px
    """
    # Target spacing: If not transmitted, use the main target.
    tgt_leg_torso = target_gap_px if tgt_leg_torso is None else tgt_leg_torso
    tgt_arm_torso = 0.7*target_gap_px if tgt_arm_torso is None else tgt_arm_torso
    tgt_leg_leg   = 0.8*target_gap_px if tgt_leg_leg   is None else tgt_leg_leg
    tgt_arm_arm   = 0.7*target_gap_px if tgt_arm_arm   is None else tgt_arm_arm
    tgt_arm_leg   = 0.8*target_gap_px if tgt_arm_leg   is None else tgt_arm_leg

    C = build_capsules_px(points_2d, r_px_by_seg)
    terms, argmins = silhouette_terms_from_capsules_v2(C)

    # —— Still following the “non-LoA leg” strategy, select the side of leg_torso to score. —— 
    move = loa_leg_side(selected_loa)
    if move is None:
        cand = { 'L': terms['torso_leg_L'], 'R': terms['torso_leg_R'] }
        exist = [(k,v) for k,v in cand.items() if v is not None]
        move = (min(exist, key=lambda kv: kv[1])[0]) if exist else None
    else:
        move = ('R' if move=='L' else 'L')

    sc_leg_torso = 0.0
    if move == 'L' and terms['torso_leg_L'] is not None:
        sc_leg_torso = gap_reward(terms['torso_leg_L'], tgt_leg_torso)
    elif move == 'R' and terms['torso_leg_R'] is not None:
        sc_leg_torso = gap_reward(terms['torso_leg_R'], tgt_leg_torso)

    # —— Other light weighting items: Add a point of reward/penalty as long as they exist. —— 
    sc_arm_torso = 0.0
    for s in ('L','R'):
        v = terms[f'torso_arm_{s}']
        if v is not None:
            sc_arm_torso += 0.5 * gap_reward(v, tgt_arm_torso)  # 0.5 on each side

    sc_leg_leg = 0.0 if terms['leg_leg'] is None else gap_reward(terms['leg_leg'], tgt_leg_leg)
    sc_arm_arm = 0.0 if terms['arm_arm'] is None else gap_reward(terms['arm_arm'], tgt_arm_arm)
    sc_arm_leg = 0.0 if terms['arm_leg'] is None else gap_reward(terms['arm_leg'], tgt_arm_leg)

    sc_keep = 0.0
    if keep_ctrl is not None and selected_loa:
        import score
        sc_keep, _, _ = score.line_of_action_score(selected_loa, keep_ctrl)

    score_total = (w_leg_torso  * sc_leg_torso +
                   w_arm_torso  * sc_arm_torso +
                   w_leg_leg    * sc_leg_leg   +
                   w_arm_arm    * sc_arm_arm   +
                   w_arm_leg    * sc_arm_leg   +
                   w_keep       * sc_keep)

    return float(score_total), terms, argmins, move

"-------------------------1) 3D capsule self-collision （world-space）----------------------------"
def _segseg_dist3d(A0, A1, B0, B1):
    """
    3D line segment-line segment shortest distance (closed-form solution), returns d (scalar, >=0).
    Reference: Christer Ericson, Real-Time Collision Detection (simplified boundary handling).
    """
    u = A1 - A0; v = B1 - B0; w0 = A0 - B0
    a = np.dot(u,u) + 1e-12
    b = np.dot(u,v)
    c = np.dot(v,v) + 1e-12
    d = np.dot(u,w0)
    e = np.dot(v,w0)
    D = a*c - b*b
    sc = 0.0; tc = 0.0
    if D > 1e-12:
        sc = (b*e - c*d) / D
        tc = (a*e - b*d) / D
    sc = min(1.0, max(0.0, sc))
    tc = min(1.0, max(0.0, tc))
    Pc = A0 + sc * u
    Qc = B0 + tc * v
    return float(np.linalg.norm(Pc - Qc))

def _capsule_radius_world(rw, t01=None):
    """
    Take the “world radius” (cone ends r0/r1; constant radius r) from the radii_world entry.
    If t01 (0~1) is given, linear interpolation can be performed; if not passed, the maximum 
        value of the two ends is returned (for safety).
    """
    if rw['type'] == 'cone':
        if t01 is None:
            return max(float(rw['r0']), float(rw['r1']))
        return (1.0 - t01) * float(rw['r0']) + t01 * float(rw['r1'])
    return float(rw['r'])

def _get_seg_AB(pose_globals, j0, j1):
    return pose_globals[j0][:3,3], pose_globals[j1][:3,3]

def collision_penalty_capsule3d(pose_globals, radii_world, pairs=None, topk=4, tgt_mm=20.0):
    """
    Perform 3D capsule separation only on a “small number of key pairs”: sep3d = d3d - (rA+rB).
    Return: non-negative penalty (only >0 for overlap/too close), higher value → worse.
    """
    if pairs is None:
        pairs = [
            ('torso', 'thigh_L'), ('torso', 'calf_L'),
            ('torso', 'thigh_R'), ('torso', 'calf_R'),
            ('torso', 'upperarm_L'), ('torso','forearm_L'),
            ('torso', 'upperarm_R'), ('torso','forearm_R'),
            ('thigh_L','thigh_R'),
            ('forearm_L','forearm_R'), 
            ('upperarm_L','upperarm_R') 
        ]
    vals = []
    for sa, sb in pairs:
        if sa not in radii_world or sb not in radii_world: 
            continue
        ja0, ja1, _ = SEG_DEFS[sa]
        jb0, jb1, _ = SEG_DEFS[sb]
        if ja0 not in pose_globals or ja1 not in pose_globals: 
            continue
        if jb0 not in pose_globals or jb1 not in pose_globals: 
            continue
        A0, A1 = _get_seg_AB(pose_globals, ja0, ja1)
        B0, B1 = _get_seg_AB(pose_globals, jb0, jb1)

        d = _segseg_dist3d(A0, A1, B0, B1)
        rA = _capsule_radius_world(radii_world[sa])
        rB = _capsule_radius_world(radii_world[sb])
        sep = d - (rA + rB)
        # Only convert “too close” items into positive penalties (unitization).
        pen = max(0.0, (0.0 - sep) / max(1e-6, tgt_mm))  # sep<0 → punishment > 0
        vals.append(pen)

    if not vals:
        return 0.0
    # Take the worst top-k overlaps (to avoid a bunch of small neighbor noise).
    vals.sort(reverse=True)
    return float(sum(vals[:topk]))

"-------------------------2) Ground/foot contact guardrail----------------------------"
def detect_grounded_feet_from_orig(pose_globals_orig, ground_y=None, eps=0.02):
    """
    Identify which foot is on the ground from the “original pose.”
    - Prioritize Toe_* / Foot_*; if neither is available, use Ankle_*'s y (more conservative).
    - ground_y: If you can transfer the ground height from Maya, that's best; otherwise, estimate the minimum y.
    Return: {‘L’: True/False, ‘R’: True/False}, ground_y
    """
    candL = [k for k in ('ToesEnd_L','Toes_L','Ankle_L') if k in pose_globals_orig]
    candR = [k for k in ('ToesEnd_R','Toes_R','Ankle_R') if k in pose_globals_orig]
    ys = []
    if ground_y is None:
        for k in candL + candR:
            ys.append(pose_globals_orig[k][:3,3][1])
        ground_y = min(ys) if ys else 0.0

    def _on_ground(cands):
        if not cands: 
            return False
        y = min(pose_globals_orig[cands[0]][:3,3][1], pose_globals_orig[cands[-1]][:3,3][1])
        return (abs(y - ground_y) <= eps)

    return {'L': _on_ground(candL), 'R': _on_ground(candR)}, float(ground_y)

def ground_penalty(pose_globals_now, grounded, ground_y, up_vec_thresh_deg=25.0):
    """
    Create two guardrails for the foot that “should touch the ground”:
    1) The sole of the foot must not touch the ground: y_toe/foot >= ground_y
    2) The sole of the foot should be as close to the ground normal as possible (approximation: the angle 
        between the direction from the ankle to the toe and +Y should be small)
    Return a non-negative penalty.
    """
    pen = 0.0
    Y = np.array([0.0,1.0,0.0], float)

    def _get_joint(side, names):
        for n in names:
            k = f'{n}_{side}'
            if k in pose_globals_now:
                return pose_globals_now[k][:3,3]
        return None

    for side in ('L','R'):
        if not grounded.get(side, False):
            continue
        toe = _get_joint(side, ['ToesEnd','Toes','Ankle'])
        ank = _get_joint(side, ['Ankle','Toes','ToesEnd'])
        if toe is None or ank is None:
            continue

        # High guardrail: linear deduction for crossing the ground
        y = float(toe[1])
        if y < ground_y:
            pen += (ground_y - y) * 50.0  # 50≈Convert rice into a tangible quantity

        # Towards the guardrail: Ankle→Toe and angle with +Y
        v = toe - ank
        nv = np.linalg.norm(v)
        if nv > 1e-6:
            cosang = float(np.dot(v/nv, Y))
            ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
            if ang > up_vec_thresh_deg:
                pen += (ang - up_vec_thresh_deg) / up_vec_thresh_deg
    return float(pen)

"-------------------------3) Hand adhesion maintenance (automatic detection & continued adhesion)----------------------------"
def _point_seg_dist3d(P, A0, A1):
    """Point to line segment 3D closest distance d, and parameter t∈[0,1]."""
    v = A1 - A0
    vv = float(np.dot(v,v)) + 1e-12
    t = float(np.dot(P - A0, v) / vv)
    t = min(1.0, max(0.0, t))
    proj = A0 + t * v
    return float(np.linalg.norm(P - proj)), t

def detect_hand_contacts_AB(pose_globals_AB, radii_world, band_mm=0.1):
    """
    Check whether Wrist_L/R is “in contact” with a segment (torso/thigh_/calf_) based on the A+B result.  
    If minsep3d ∈ [-band, +band], it is considered contact; return anchors:
       {‘Wrist_L’: (‘torso’ or ‘thigh_L’ etc.), ‘Wrist_R’: ...} or omit the key to indicate no contact.
    band_mm: This is the tolerance threshold for “the hand being considered to be touching/adjacent to a certain part.”
    """
    anchors = {}
    cand_targets = ['torso','thigh_L','calf_L','thigh_R','calf_R']
    for side in ('L','R'):
        wname = f'Wrist_{side}'
        if wname not in pose_globals_AB:
            continue
        Pw = pose_globals_AB[wname][:3,3]
        best_sep = None; best_seg = None
        for seg in cand_targets:
            if seg not in radii_world: 
                continue
            j0, j1, _ = SEG_DEFS[seg]
            if j0 not in pose_globals_AB or j1 not in pose_globals_AB:
                continue
            A0, A1 = _get_seg_AB(pose_globals_AB, j0, j1)
            d, t = _point_seg_dist3d(Pw, A0, A1)
            rw = _capsule_radius_world(radii_world[seg], t01=t)
            sep = d - rw
            if (best_sep is None) or (sep < best_sep):
                best_sep, best_seg = sep, seg
        if (best_sep is not None) and (abs(best_sep) <= band_mm):
            anchors[wname] = best_seg
    return anchors

def hand_contact_penalty(pose_globals_now, radii_world, anchors, band_mm=0.1, soft=0.2):
    """
    Apply a band penalty to the wrist identified as being in contact: desired sep3d ∈ [0, band_mm] 
        (adhering to the surface/slightly detached).
    sep3d = d(point, segment) - r(segment_at_t)
    Penalty = distance deviating from the outer side of the band / band_mm
    """
    pen = 0.0
    for wname, seg in anchors.items():
        if (wname not in pose_globals_now) or (seg not in radii_world):
            continue
        Pw = pose_globals_now[wname][:3,3]
        j0, j1, _ = SEG_DEFS[seg]
        if j0 not in pose_globals_now or j1 not in pose_globals_now:
            continue
        A0, A1 = _get_seg_AB(pose_globals_now, j0, j1)
        d, t = _point_seg_dist3d(Pw, A0, A1)
        rw = _capsule_radius_world(radii_world[seg], t01=t)
        sep = d - rw  # <0 breakthrough
        # 希望 0 <= sep <= band_mm：Penalties only apply outside the interval
        if sep < 0.0:
            pen += (-sep) / band_mm
        elif sep > band_mm:
            pen += (sep - band_mm) / band_mm * soft
    return float(pen)


"===================================Silhouette Part End =================================="
  
def smooth_penalty(rot_now, rot_orig, lam=1e-3):
    """
    rot_* : dict{joint:[x,y,z]}, calculates the L2 sum of all joint Euler differences
    """
    acc = 0.0
    for j, r0 in rot_orig.items():
        r1 = rot_now.get(j)
        if r1 is not None:
            acc += np.sum((np.asarray(r1)-np.asarray(r0))**2)
    return -lam*acc

def _rot_geodesic_deg(R_now, R_ref):
    """
     Returns the geometric distance (angle, degrees) between two rotation matrices.
    """
    R_rel = R_ref.T @ R_now
    # 数值稳定：trace 可能略超出 [-1,3]
    cos_theta = (np.trace(R_rel) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))
    return float(theta)

def se3_penalty(T_now, T_ref, w_t=1.0, w_r=1.0):
    """
    For a single joint: weighted sum of translation L2 + rotation geometric angle (as “cost”).
    Return positive numbers (large = large deviation). The outer layer takes negative values as a penalty.
    """
    t_now = T_now[:3, 3]; t_ref = T_ref[:3, 3]
    R_now = T_now[:3, :3]; R_ref = T_ref[:3, :3]
    trans_err = np.linalg.norm(t_now - t_ref)
    rot_err   = _rot_geodesic_deg(R_now, R_ref)
    return w_t * trans_err + w_r * rot_err

def rigid_penalty_global_v2(T_now, T_ref, *,
                            constraints=None,
                            use_toe_anchor=False,
                            toe_names=('Toe_L','Toe_R')):
    """
    constraints: dict[joint_name] = dict(w_t=..., w_r=...)
    constraints: dict[joint_name] = dict(w_t=..., w_r=...)
      w_t: Translation error weight (unit ~ cm)
      w_r: Rotation error weight (unit ~ degrees)
    use_toe_anchor: Whether to give Toe a light weight to stabilize the foot
    Return: Negative number (The more negative, the greater the penalty)
    """
    if constraints is None:
        # Default set
        constraints = {
            'HeadEnd_M':   dict(w_t=0.004, w_r=0.0),

            'Wrist_L':  dict(w_t=0.002, w_r=0.001),
            'Wrist_R':  dict(w_t=0.002, w_r=0.001),

            'Ankle_L':  dict(w_t=0.005, w_r=0.02),
            'Ankle_R':  dict(w_t=0.005, w_r=0.02),
        }

    loss = 0.0
    for jn, w in constraints.items():
        if (jn in T_now) and (jn in T_ref):
            loss += se3_penalty(T_now[jn], T_ref[jn], w_t=w.get('w_t', 0.0), w_r=w.get('w_r', 0.0))

    if use_toe_anchor:
        # Add “light constraints” to the toes to help stabilize the foot, but don't too high or it will affect footwork performance.
        for jn in toe_names:
            if (jn in T_now) and (jn in T_ref):
                loss += se3_penalty(T_now[jn], T_ref[jn], w_t=0.0015, w_r=0.008)

    return -float(loss)  # As punishment, take the negative sign.
