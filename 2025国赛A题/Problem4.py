import math
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from itertools import permutations

g = 9.81
VM = 300.0
V_SINK = 3.0
R_SMOKE = 10.0
T_EFFECT = 20.0


R_TAR, H_TAR = 7.0, 10.0
CYL_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)
P_TARGET = np.array([0.0, 200.0, 5.0], dtype=float)
M0 = np.array([20000.0, 0.0, 2000.0], dtype=float)

U0_FY1 = np.array([17800.0,     0.0, 1800.0], dtype=float)
U0_FY2 = np.array([12000.0,  1400.0, 1400.0], dtype=float)
U0_FY3 = np.array([ 6000.0, -3000.0,  700.0], dtype=float)
UAVS = [
    {"name": "FY1", "U0": U0_FY1},
    {"name": "FY2", "U0": U0_FY2},
    {"name": "FY3", "U0": U0_FY3},
]

def missile_hit_time(m0: np.ndarray) -> float:
    return float(np.linalg.norm(m0) / VM)

T_HIT = missile_hit_time(M0)

# 角度显示归一化
def deg360(theta_rad: float) -> float:
    d = math.degrees(theta_rad)
    d = (d % 360.0 + 360.0) % 360.0
    return 0.0 if abs(d-360.0) < 1e-9 else d


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v

def missile_pos(m0: np.ndarray, t: float) -> np.ndarray:
    d = unit(-m0)
    return m0 + VM * d * t

def uav_pos(u0: np.ndarray, theta: float, v_u: float, t: float) -> np.ndarray:
    hx, hy = math.cos(theta), math.sin(theta)
    return np.array([u0[0] + v_u * hx * t, u0[1] + v_u * hy * t, u0[2]], dtype=float)

def burst_point(u0: np.ndarray, theta: float, v_u: float, t_drop: float, tau: float) -> np.ndarray:
    hx, hy = math.cos(theta), math.sin(theta)
    r_drop = uav_pos(u0, theta, v_u, t_drop)
    horiz  = np.array([v_u * hx * tau, v_u * hy * tau, 0.0], dtype=float)
    vert   = np.array([0.0, 0.0, -0.5 * g * tau * tau], dtype=float)
    return r_drop + horiz + vert

def smoke_center_after_burst(s_burst: np.ndarray, t: float, t_burst: float) -> np.ndarray:
    dz = -V_SINK * max(0.0, t - t_burst)
    return s_burst + np.array([0.0, 0.0, dz], dtype=float)

def point_to_segment_dist(P: np.ndarray, Q: np.ndarray, X: np.ndarray) -> float:
    v = Q - P
    vv = float(np.dot(v, v))
    if vv == 0.0:
        return float(np.linalg.norm(X - P))
    a = float(np.dot(X - P, v) / vv)
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    Y = P + a * v
    return float(np.linalg.norm(X - Y))

def covered_L0_at_time(m0, p_target, s_burst, t_burst, t) -> bool:
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    return (point_to_segment_dist(p_target, m_t, s_t) <= R_SMOKE)

def clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def cyl_points_top_bottom(N_ang: int = 48) -> np.ndarray:
    cx, cy, cz = CYL_CENTER
    pts = []
    for z in (cz, cz + H_TAR):
        for k in range(N_ang):
            ang = 2.0 * math.pi * k / N_ang
            pts.append((cx + R_TAR * math.cos(ang), cy + R_TAR * math.sin(ang), z))
    return np.array(pts, dtype=float)

def cyl_points_side(N_ang: int = 48, N_Z: int = 9) -> np.ndarray:
    cx, cy, cz = CYL_CENTER
    zs = np.linspace(cz, cz + H_TAR, N_Z)
    pts = []
    for z in zs:
        for k in range(N_ang):
            ang = 2.0 * math.pi * k / N_ang
            pts.append((cx + R_TAR * math.cos(ang), cy + R_TAR * math.sin(ang), z))
    return np.array(pts, dtype=float)

def build_cylinder_samples(N_ang=48, N_z=9, include_side=True) -> np.ndarray:
    pts = [cyl_points_top_bottom(N_ang)]
    if include_side:
        pts.append(cyl_points_side(N_ang, N_z))
    return np.concatenate(pts, axis=0)

def covered_L1_at_time_vectorized(m0, s_burst, t_burst, t, PTS) -> bool:
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    v = m_t - PTS
    w = s_t - PTS
    vv = np.sum(v * v, axis=1)
    alpha = np.divide(np.sum(w * v, axis=1), vv, out=np.zeros_like(vv), where=vv > 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    Y = PTS + alpha[:, None] * v
    dist = np.linalg.norm(s_t - Y, axis=1)
    return bool(np.any(dist <= R_SMOKE))


# 图论：候选生成 + 掩码 + 贪心 + 兜底

def _time_grid(dt: float = 0.02) -> np.ndarray:
    return np.arange(0.0, T_HIT + 1e-12, dt)

def _candidate_from_anchor(u0: np.ndarray, frac: float, alpha: float, tau_mult: float, clamp_eps: float = 0.10):
    """
    几何锚点生成单个候选（θ, v, t_drop, τ）
    """
    t_b = clip(frac * min(60.0, T_HIT - 2.0), 0.5, 59.5)
    m_tb = missile_pos(M0, t_b)
    Y = P_TARGET + alpha * (m_tb - P_TARGET)

    dx, dy = Y[0] - u0[0], Y[1] - u0[1]
    theta = math.atan2(dy, dx)
    D_xy = math.hypot(dx, dy)
    v = clip(D_xy / t_b, 70.0, 140.0)

    u0z, Yz = float(u0[2]), float(Y[2])
    if u0z > Yz:
        tau_base = math.sqrt(max(0.0, 2.0 * (u0z - Yz)) / g)
    else:
        tau_base = 0.2

    tau_max_by_t = max(0.2, t_b - clamp_eps)
    tau = clip(tau_base * tau_mult, 0.2, min(12.0, tau_max_by_t))
    t_drop = t_b - tau
    return theta % (2.0 * math.pi), v, t_drop, tau

def _candidate_mask_L0(uav_idx: int, theta: float, v: float, t_drop: float, tau: float,
                       tgrid: np.ndarray) -> np.ndarray:
    """生成某候选在 L0 判定下的时间掩码"""
    u0 = UAVS[uav_idx]["U0"]
    t_burst = t_drop + tau
    if t_burst >= T_HIT:
        return np.zeros_like(tgrid, dtype=bool)
    s_burst = burst_point(u0, theta, v, t_drop, tau)
    if s_burst[2] <= 0.0:
        return np.zeros_like(tgrid, dtype=bool)

    mask = np.zeros_like(tgrid, dtype=bool)
    t_start, t_end = t_burst, min(t_burst + T_EFFECT, T_HIT)
    if t_end <= t_start:
        return mask

    idx = np.where((tgrid >= t_start) & (tgrid <= t_end))[0]
    for k in idx:
        t = float(tgrid[k])
        if covered_L0_at_time(M0, P_TARGET, s_burst, t_burst, t):
            mask[k] = True
    return mask

def _candidate_mask_L1(uav_idx: int, theta: float, v: float, t_drop: float, tau: float,
                       tgrid: np.ndarray, PTS: np.ndarray) -> np.ndarray:
    """生成某候选在 L1 判定下的时间掩码"""
    u0 = UAVS[uav_idx]["U0"]
    t_burst = t_drop + tau
    if t_burst >= T_HIT:
        return np.zeros_like(tgrid, dtype=bool)
    s_burst = burst_point(u0, theta, v, t_drop, tau)
    if s_burst[2] <= 0.0:
        return np.zeros_like(tgrid, dtype=bool)

    mask = np.zeros_like(tgrid, dtype=bool)
    t_start, t_end = t_burst, min(t_burst + T_EFFECT, T_HIT)
    if t_end <= t_start:
        return mask

    idx = np.where((tgrid >= t_start) & (tgrid <= t_end))[0]
    for k in idx:
        t = float(tgrid[k])
        if covered_L1_at_time_vectorized(M0, s_burst, t_burst, t, PTS):
            mask[k] = True
    return mask

def build_candidates_q4(
    fracs = (0.12, 0.25, 0.40, 0.55, 0.70, 0.85, 0.92, 0.96),
    alphas = (0.60, 0.70, 0.80, 0.88, 0.92, 0.96, 0.985),
    taus   = (0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50),
    per_uav_keep: int = 28,
    dt_mask: float = 0.015,
    mask_mode: str = 'L0',              # 'L0' | 'L1'
    PTS: Optional[np.ndarray] = None     # L1时必需
) -> Tuple[List[Dict[str, Any]], np.ndarray]:

    tgrid = _time_grid(dt_mask)
    all_cands = []
    for uav_idx in range(3):
        local = []
        u0 = UAVS[uav_idx]["U0"]
        for f in fracs:
            for a in alphas:
                for tau_mult in taus:
                    th, v, td, ta = _candidate_from_anchor(u0, f, a, tau_mult)
                    if mask_mode.upper() == 'L1':
                        if PTS is None:
                            raise ValueError("L1 评估需要提供采样点 PTS")
                        mask = _candidate_mask_L1(uav_idx, th, v, td, ta, tgrid, PTS)
                    else:
                        mask = _candidate_mask_L0(uav_idx, th, v, td, ta, tgrid)
                    score = float(mask.sum() * dt_mask)
                    if score > 0.0:
                        local.append({
                            "uav": uav_idx,
                            "theta": th, "v": v, "t_drop": td, "tau": ta,
                            "mask": mask, "score": score
                        })
        local.sort(key=lambda x: x["score"], reverse=True)
        all_cands += local[:per_uav_keep]
    return all_cands, tgrid

def remask_candidates(
    candidates: List[Dict[str, Any]],
    tgrid: np.ndarray,
    mask_mode: str,
    dt_mask: float,
    PTS: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """把候选重新用指定模式打分"""
    out = []
    for c in candidates:
        u = c["uav"]; th, v, td, ta = c["theta"], c["v"], c["t_drop"], c["tau"]
        if mask_mode.upper() == 'L1':
            if PTS is None:
                raise ValueError("L1 评估需要提供采样点 PTS")
            mask = _candidate_mask_L1(u, th, v, td, ta, tgrid, PTS)
        else:
            mask = _candidate_mask_L0(u, th, v, td, ta, tgrid)
        sc = float(mask.sum() * dt_mask)
        if sc <= 0.0:
            continue
        c2 = dict(c)
        c2["mask"] = mask
        c2["score"] = sc
        out.append(c2)
    return out

# 兜底：寻找最大未覆盖时间空档
def _largest_gap(mask: np.ndarray, tgrid: np.ndarray) -> Tuple[float, float]:
    if len(tgrid) < 2:
        return 0.0, 0.0
    gaps = []
    in_gap = False; a = None
    for k in range(len(mask)):
        if (not mask[k]) and (not in_gap):
            in_gap = True; a = float(tgrid[k])
        if in_gap and (k == len(mask)-1 or mask[k+1]):
            b = float(tgrid[k])
            gaps.append((a, b))
            in_gap = False
    if not gaps:
        return float(tgrid[len(tgrid)//2]), 0.0
    a, b = max(gaps, key=lambda it: it[1]-it[0])
    return 0.5*(a+b), (b-a)

def _synthesize_and_polish_for_gap(uav_idx: int, gap_center_t: float,
                                   tgrid: np.ndarray,
                                   mask_mode: str = 'L0',
                                   PTS: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
    """围绕最大空档时间点合成一批邻域候选，选择得分最高者兜底"""
    u0 = UAVS[uav_idx]["U0"]
    t_b = clip(gap_center_t, 0.6, min(59.5, T_HIT-0.5))
    frac = t_b / min(60.0, T_HIT-2.0)
    alphas_try = (0.92, 0.96, 0.985)
    tau_mult_try = (0.70, 0.85, 1.00, 1.15)
    dt = float(tgrid[1]-tgrid[0])

    def mk(theta, v, td, ta):
        if mask_mode.upper() == 'L1':
            if PTS is None:
                return None
            mask = _candidate_mask_L1(uav_idx, theta, v, td, ta, tgrid, PTS)
        else:
            mask = _candidate_mask_L0(uav_idx, theta, v, td, ta, tgrid)
        return {"uav": uav_idx, "theta":theta, "v":v, "t_drop":td, "tau":ta,
                "mask":mask, "score": float(mask.sum()*dt)}

    best = None
    for a in alphas_try:
        th, v, td, ta = _candidate_from_anchor(u0, frac, a, 1.0)
        neigh_yaw = [th + math.radians(d) for d in (-8,-4,0,4,8)]
        neigh_v   = [clip(v + dv, 70.0, 140.0) for dv in (-12,-6,0,6,12)]
        neigh_td  = [clip(td + d, 0.0, 60.0) for d in (-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5)]
        neigh_tau = [clip(ta*m, 0.2, min(12.0, t_b-0.1)) for m in tau_mult_try]
        for th1 in neigh_yaw:
            for v1 in neigh_v:
                for td1 in neigh_td:
                    for ta1 in neigh_tau:
                        c1 = mk(th1, v1, td1, ta1)
                        if c1 is None:
                            continue
                        if best is None or c1["score"] > best["score"]:
                            best = c1
    return best

# 每个 UAV 恰好选 1
def select_exact_one_per_uav(
    candidates: List[Dict[str, Any]],
    tgrid: np.ndarray,
    mask_mode: str = 'L0',
    PTS: Optional[np.ndarray] = None
) -> Tuple[List[Dict[str, Any]], float, np.ndarray]:
    """分区拟阵 + 兜底"""
    groups = {0: [], 1: [], 2: []}
    for c in candidates:
        groups[c["uav"]].append(c)
    dt = float(tgrid[1] - tgrid[0])

    best_score, best_sol, best_mask = -1.0, None, None
    for order in permutations([0, 1, 2]):
        union_mask = np.zeros_like(tgrid, dtype=bool)
        picked = []
        feasible = True
        for u in order:
            best_c, best_gain = None, -1.0
            for c in groups[u]:
                new_cover = np.logical_or(union_mask, c["mask"])
                gain = float((new_cover.sum() - union_mask.sum()) * dt)
                if gain > best_gain:
                    best_gain, best_c = gain, c
            if best_c is None or best_gain <= 0.0:
                center_t, _ = _largest_gap(union_mask, tgrid)
                synth = _synthesize_and_polish_for_gap(u, center_t, tgrid, mask_mode=mask_mode, PTS=PTS)
                if synth is not None:
                    new_cover = np.logical_or(union_mask, synth["mask"])
                    gain = float((new_cover.sum() - union_mask.sum()) * dt)
                    if gain > best_gain:
                        best_gain, best_c = gain, synth
            if best_c is None:
                feasible = False
                break
            picked.append(best_c)
            union_mask = np.logical_or(union_mask, best_c["mask"])
        if feasible:
            total = float(union_mask.sum() * dt)
            if total > best_score:
                best_score, best_sol, best_mask = total, picked, union_mask

    if best_sol is None:
        return [], 0.0, np.zeros_like(tgrid, dtype=bool)
    return best_sol, best_score, best_mask

def greedy_partition_matroid_max_coverage(
    candidates: List[Dict[str, Any]],
    tgrid: np.ndarray,
    per_uav_limit: int = 1,
    K_total: int = 3
) -> Tuple[List[Dict[str, Any]], float, np.ndarray]:
    """备用贪心（不强制每 UAV 必选）"""
    chosen = []
    union_mask = np.zeros_like(tgrid, dtype=bool)
    counts = {0: 0, 1: 0, 2: 0}
    dt = float(tgrid[1] - tgrid[0])

    while len(chosen) < K_total:
        best_gain, best_c = 0.0, None
        for c in candidates:
            u = c["uav"]
            if counts[u] >= per_uav_limit:
                continue
            new_cover = np.logical_or(union_mask, c["mask"])
            gain = float((new_cover.sum() - union_mask.sum()) * dt)
            if gain > best_gain:
                best_gain, best_c = gain, c
        if best_c is None or best_gain <= 0.0:
            break
        chosen.append(best_c)
        union_mask = np.logical_or(union_mask, best_c["mask"])
        counts[best_c["uav"]] += 1
        candidates = [x for x in candidates if x is not best_c]

    total = float(union_mask.sum() * dt)
    return chosen, total, union_mask

def _mask_to_intervals(mask: np.ndarray, tgrid: np.ndarray) -> List[Tuple[float, float]]:
    """把布尔时间掩码转换为区间列表"""
    intervals = []
    if len(mask) == 0:
        return intervals
    in_seg = False
    a = None
    for k in range(len(mask)):
        if mask[k] and not in_seg:
            in_seg, a = True, float(tgrid[k])
        if in_seg and (k == len(mask) - 1 or (not mask[k + 1])):
            b = float(tgrid[k])
            intervals.append((round(a, 3), round(b, 3)))
            in_seg = False
    return intervals


def solve_q4_graph(
    strategy: str = 'L0',          # 'L0' | 'L1' | 'DUAL'
    dt_mask: float = 0.015,
    # 候选池密度
    fracs=(0.12, 0.25, 0.40, 0.55, 0.70, 0.85, 0.92, 0.96, 0.985, 0.995),
    alphas=(0.60, 0.70, 0.80, 0.88, 0.92, 0.96, 0.985),
    taus=(0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50),
    per_uav_keep=28,
    N_ANG: int = 48, N_Z: int = 9, INCLUDE_SIDE: bool = True,
    # 选择策略
    force_one_per_uav: bool = True,
    dual_per_uav_keep_L1: Optional[int] = None,
    debug: bool = False
) -> Dict[str, Any]:

    mode = strategy.upper()
    if mode not in ('L0','L1','DUAL'):
        raise ValueError("strategy 需为 'L0' / 'L1' / 'DUAL'")

    PTS = None
    if mode in ('L1','DUAL'):
        PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)

    # 单策略：L0 或 L1
    if mode in ('L0','L1'):
        cands, tgrid = build_candidates_q4(
            fracs=fracs, alphas=alphas, taus=taus,
            per_uav_keep=per_uav_keep, dt_mask=dt_mask,
            mask_mode=mode, PTS=PTS
        )
        if debug:
            counts = {i: sum(1 for c in cands if c["uav"]==i) for i in range(3)}
            print(f"[debug] mask_mode={mode}, per-UAV nonzero candidates:", counts)

        if force_one_per_uav:
            picked, cover_val, union_mask = select_exact_one_per_uav(cands, tgrid, mask_mode=mode, PTS=PTS)
        else:
            picked, cover_val, union_mask = greedy_partition_matroid_max_coverage(
                candidates=cands, tgrid=tgrid, per_uav_limit=1, K_total=3
            )

        ths = [p["theta"] for p in picked]
        vs  = [p["v"] for p in picked]
        tds = [p["t_drop"] for p in picked]
        taus_= [p["tau"] for p in picked]
        uidx = [p["uav"] for p in picked]
        unames = [UAVS[u]["name"] for u in uidx]

        bursts = []
        for p in picked:
            u0 = UAVS[p["uav"]]["U0"]
            t_burst = p["t_drop"] + p["tau"]
            sb = burst_point(u0, p["theta"], p["v"], p["t_drop"], p["tau"])
            bursts.append({
                "uav": UAVS[p["uav"]]["name"],
                "t_drop": float(p["t_drop"]),
                "tau": float(p["tau"]),
                "t_burst": float(t_burst),
                "s_burst": (float(sb[0]), float(sb[1]), float(sb[2]))
            })
        bursts = sorted(bursts, key=lambda b: b["t_burst"])
        intervals = _mask_to_intervals(union_mask, tgrid)

        return {
            "method": "graph_max_coverage",
            "strategy": mode,
            "theta_deg": [round(deg360(x), 3) for x in ths],
            "v_u_mps": [round(x, 3) for x in vs],
            "drops_s": [round(x, 3) for x in tds],
            "taus_s":  [round(x, 3) for x in taus_],
            "uav_names": unames,
            "uav_indices": uidx,
            "bursts":  bursts,
            "cover_total_s": cover_val,
            "cover_intervals_s": intervals,
            "config": {
                "dt_mask": dt_mask, "fracs": fracs, "alphas": alphas, "taus": taus,
                "per_uav_keep": per_uav_keep, "force_one_per_uav": force_one_per_uav,
                **({"N_ANG": N_ANG, "N_Z": N_Z, "INCLUDE_SIDE": INCLUDE_SIDE} if mode!='L0' else {})
            }
        }

    # 双策略

    # 1. 用 L0 快速建池与初筛
    cands_L0, tgrid = build_candidates_q4(
        fracs=fracs, alphas=alphas, taus=taus,
        per_uav_keep=per_uav_keep, dt_mask=dt_mask,
        mask_mode='L0', PTS=None
    )
    if debug:
        counts0 = {i: sum(1 for c in cands_L0 if c["uav"]==i) for i in range(3)}
        print("[debug][DUAL] L0 pool per-UAV:", counts0)

    # 2. 对 L0 池“换评估模式”为 L1 并重新打分
    PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)
    cands_L1 = remask_candidates(cands_L0, tgrid, 'L1', dt_mask, PTS=PTS)
    # 截断每UAV数量
    if dual_per_uav_keep_L1 is None:
        dual_per_uav_keep_L1 = per_uav_keep
    groups = {0: [], 1: [], 2: []}
    for c in cands_L1: groups[c["uav"]].append(c)
    cands_L1_trim = []
    for u in [0,1,2]:
        groups[u].sort(key=lambda x: x["score"], reverse=True)
        cands_L1_trim += groups[u][:dual_per_uav_keep_L1]
    if debug:
        counts1 = {i: sum(1 for c in cands_L1_trim if c["uav"]==i) for i in range(3)}
        print("[debug][DUAL] L1 re-score pool per-UAV:", counts1)

    # 3. 在 L1 掩码下正式选择
    if force_one_per_uav:
        picked, cover_val, union_mask = select_exact_one_per_uav(cands_L1_trim, tgrid, mask_mode='L1', PTS=PTS)
    else:
        picked, cover_val, union_mask = greedy_partition_matroid_max_coverage(
            candidates=cands_L1_trim, tgrid=tgrid, per_uav_limit=1, K_total=3
        )

    ths = [p["theta"] for p in picked]
    vs  = [p["v"] for p in picked]
    tds = [p["t_drop"] for p in picked]
    taus_= [p["tau"] for p in picked]
    uidx = [p["uav"] for p in picked]
    unames = [UAVS[u]["name"] for u in uidx]

    bursts = []
    for p in picked:
        u0 = UAVS[p["uav"]]["U0"]
        t_burst = p["t_drop"] + p["tau"]
        sb = burst_point(u0, p["theta"], p["v"], p["t_drop"], p["tau"])
        bursts.append({
            "uav": UAVS[p["uav"]]["name"],
            "t_drop": float(p["t_drop"]),
            "tau": float(p["tau"]),
            "t_burst": float(t_burst),
            "s_burst": (float(sb[0]), float(sb[1]), float(sb[2]))
        })
    bursts = sorted(bursts, key=lambda b: b["t_burst"])
    intervals = _mask_to_intervals(union_mask, tgrid)

    return {
        "method": "graph_max_coverage",
        "strategy": "DUAL(L0→L1)",
        "theta_deg": [round(deg360(x), 3) for x in ths],
        "v_u_mps": [round(x, 3) for x in vs],
        "drops_s": [round(x, 3) for x in tds],
        "taus_s":  [round(x, 3) for x in taus_],
        "uav_names": unames,             # ← 供重建/校验
        "uav_indices": uidx,             # ← 供重建/校验
        "bursts":  bursts,
        "cover_total_s": cover_val,
        "cover_intervals_s": intervals,
        "config": {
                "dt_mask": dt_mask, "fracs": fracs, "alphas": alphas, "taus": taus,
                "per_uav_keep_L0": per_uav_keep, "per_uav_keep_L1": dual_per_uav_keep_L1,
                "force_one_per_uav": True,
                "N_ANG": N_ANG, "N_Z": N_Z, "INCLUDE_SIDE": INCLUDE_SIDE
        }
    }

def _reconstruct_union_from_ans(ans: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 ans 返回的参数重建并集掩码，作为一致性校验依据。
    """
    cfg = ans.get("config", {})
    dt = float(cfg.get("dt_mask", 0.015))
    tgrid = _time_grid(dt)
    strategy = str(ans.get("strategy", "L0")).upper()
    mode = 'L1' if ('L1' in strategy) else 'L0'

    # 构建 PTS
    PTS = None
    if mode == 'L1':
        PTS = build_cylinder_samples(
            N_ang=int(cfg.get("N_ANG", 48)),
            N_z=int(cfg.get("N_Z", 9)),
            include_side=bool(cfg.get("INCLUDE_SIDE", True))
        )

    uidx = ans.get("uav_indices", None)
    if uidx is None:
        uidx = [0, 1, 2]

    ths_deg = ans["theta_deg"]
    vs      = ans["v_u_mps"]
    tds     = ans["drops_s"]
    taus_   = ans["taus_s"]

    if not (len(ths_deg)==len(vs)==len(tds)==len(taus_)==len(uidx)):
        raise RuntimeError("返回数组长度不一致，无法重建 union 掩码")

    union = np.zeros_like(tgrid, dtype=bool)
    for i in range(len(uidx)):
        u = int(uidx[i])
        th = math.radians(float(ths_deg[i]))
        v  = float(vs[i])
        td = float(tds[i])
        ta = float(taus_[i])

        if mode == 'L1':
            mask = _candidate_mask_L1(u, th, v, td, ta, tgrid, PTS)
        else:
            mask = _candidate_mask_L0(u, th, v, td, ta, tgrid)
        union = np.logical_or(union, mask)
    return union, tgrid

def _intervals_from_mask(mask: np.ndarray, tgrid: np.ndarray) -> List[Tuple[float, float]]:
    return _mask_to_intervals(mask, tgrid)

def validate_q4_solution(ans: Dict[str, Any], verbose: bool = True, tol_s: float = 0.45):
    union, tgrid = _reconstruct_union_from_ans(ans)
    dt = float(tgrid[1]-tgrid[0])
    cover = float(union.sum()*dt)
    cover_ans = float(ans["cover_total_s"])
    if verbose:
        print(f"[check] reconstruct cover={cover:.3f}s vs ans={cover_ans:.3f}s (|Δ|={abs(cover-cover_ans):.3f}s)")
    assert abs(cover - cover_ans) <= max(tol_s, 2.5*dt), "覆盖时长与返回值不一致"

    inter = _intervals_from_mask(union, tgrid)
    inter_ans = ans["cover_intervals_s"]
    if verbose:
        print(f"[check] intervals reconstructed: {inter}")
        print(f"[check] intervals in ans      : {inter_ans}")
    assert len(inter) == len(inter_ans), "区间数量不一致"
    for (a,b), (A,B) in zip(inter, inter_ans):
        assert abs(a-A) <= 2.5*dt and abs(b-B) <= 2.5*dt, "区间端点不一致（离散误差之外）"


def _mode_from_strategy_str(strategy_str: str) -> str:
    s = (strategy_str or "").upper()
    return 'L1' if ('L1' in s) else 'L0'

def summarize_solution_rows(ans: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = ans.get("config", {})
    dt = float(cfg.get("dt_mask", 0.02))
    tgrid = _time_grid(dt)
    mode = _mode_from_strategy_str(str(ans.get("strategy", "L0")))

    PTS = None
    if mode == 'L1':
        N_ANG = int(cfg.get("N_ANG", 48))
        N_Z   = int(cfg.get("N_Z", 9))
        INCLUDE_SIDE = bool(cfg.get("INCLUDE_SIDE", True))
        PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)

    uidx_list = ans.get("uav_indices", [0,1,2])
    ths_deg   = ans["theta_deg"]
    vs        = ans["v_u_mps"]
    tds       = ans["drops_s"]
    taus_     = ans["taus_s"]
    if not (len(uidx_list)==len(ths_deg)==len(vs)==len(tds)==len(taus_)):
        raise RuntimeError("返回数组长度不一致，无法生成明细表。")

    rows = []
    for i in range(len(uidx_list)):
        u = int(uidx_list[i])
        name = UAVS[u]["name"]
        th_deg = float(ths_deg[i])
        th = math.radians(th_deg)
        v  = float(vs[i])
        td = float(tds[i])
        ta = float(taus_[i])

        r_drop = uav_pos(UAVS[u]["U0"], th, v, td)
        s_burst = burst_point(UAVS[u]["U0"], th, v, td, ta)

        if mode == 'L1':
            mask = _candidate_mask_L1(u, th, v, td, ta, tgrid, PTS)
        else:
            mask = _candidate_mask_L0(u, th, v, td, ta, tgrid)
        eff_time = float(mask.sum() * dt)

        rows.append({
            "uav": name,
            "theta_deg": th_deg,
            "v": v,
            "x_drop": float(r_drop[0]), "y_drop": float(r_drop[1]), "z_drop": float(r_drop[2]),
            "x_burst": float(s_burst[0]), "y_burst": float(s_burst[1]), "z_burst": float(s_burst[2]),
            "eff_time_s": eff_time
        })
    return rows

def print_q4_report_table(ans: Dict[str, Any]):
    rows = summarize_solution_rows(ans)
    header = (
        "无人机编号\t"
        "无人机运动方向\t"
        "无人机运动速度 (m/s)\t"
        "烟幕干扰弹投放点的x坐标 (m)\t"
        "烟幕干扰弹投放点的y坐标 (m)\t"
        "烟幕干扰弹投放点的z坐标 (m)\t"
        "烟幕干扰弹起爆点的x坐标 (m)\t"
        "烟幕干扰弹起爆点的y坐标 (m)\t"
        "烟幕干扰弹起爆点的z坐标 (m)\t"
        "有效干扰时长 (s)"
    )
    print(header)
    for r in rows:
        print(
            f"{r['uav']}\t"
            f"{r['theta_deg']:.3f}\t"
            f"{r['v']:.3f}\t"
            f"{r['x_drop']:.3f}\t"
            f"{r['y_drop']:.3f}\t"
            f"{r['z_drop']:.3f}\t"
            f"{r['x_burst']:.3f}\t"
            f"{r['y_burst']:.3f}\t"
            f"{r['z_burst']:.3f}\t"
            f"{r['eff_time_s']:.3f}"
        )


def _intervals_to_mask(intervals, tgrid):
    mask = np.zeros_like(tgrid, dtype=bool)
    if not intervals:
        return mask
    for (a, b) in intervals:
        idx = np.where((tgrid >= a - 1e-12) & (tgrid <= b + 1e-12))[0]
        mask[idx] = True
    return mask

def _mask_by_mode_for_params(uav_idx, theta_deg, v, t_drop, tau, tgrid, mode, PTS=None):
    th = math.radians(theta_deg)
    if mode.upper() == 'L1':
        if PTS is None:
            raise ValueError("L1 模式需要 PTS")
        return _candidate_mask_L1(uav_idx, th, v, t_drop, tau, tgrid, PTS)
    else:
        return _candidate_mask_L0(uav_idx, th, v, t_drop, tau, tgrid)

def _pick_best_pair_given_fixed_union(
    groups,
    tgrid,
    union_fixed
):
    dt = float(tgrid[1] - tgrid[0])
    u_left = sorted(groups.keys())
    assert len(u_left) == 2, "应仅剩两架 UAV"
    U, V = u_left[0], u_left[1]
    best = None
    best_union = None
    best_cover = -1.0

    for cU in groups[U]:
        for cV in groups[V]:
            uni = union_fixed
            uni = np.logical_or(uni, cU["mask"])
            uni = np.logical_or(uni, cV["mask"])
            cover = float(uni.sum() * dt)
            if cover > best_cover:
                best_cover = cover
                best = (cU, cV)
                best_union = uni
    return best, best_cover, best_union

def solve_q4_graph_fixed_fy1(
    strategy: str = 'DUAL',
    dt_mask: float = 0.02,
    fracs=(0.12, 0.25, 0.40, 0.55, 0.70, 0.85, 0.92, 0.96, 0.985, 0.995),
    alphas=(0.60, 0.70, 0.80, 0.88, 0.92, 0.96, 0.985),
    taus=(0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50),
    per_uav_keep=28,
    dual_per_uav_keep_L1: Optional[int] = None,
    N_ANG: int = 48, N_Z: int = 9, INCLUDE_SIDE: bool = True,
    debug: bool = False,
    fy1_theta_deg: Optional[float] = None,
    fy1_v: Optional[float] = None,
    fy1_t_drop: Optional[float] = None,
    fy1_tau: Optional[float] = None,
    fy1_intervals: Optional[List[Tuple[float, float]]] = None
) -> Dict[str, Any]:
    mode = strategy.upper()
    if mode not in ('L0','L1','DUAL'):
        raise ValueError("strategy 需为 'L0' / 'L1' / 'DUAL'")

    tgrid = _time_grid(dt_mask)
    PTS = None
    if mode in ('L1','DUAL'):
        PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)

    have_params = (fy1_theta_deg is not None and fy1_v is not None and
                   fy1_t_drop is not None and fy1_tau is not None)
    union_fixed = np.zeros_like(tgrid, dtype=bool)
    fy1_row_for_report = None

    if have_params:
        eval_mode = 'L1' if (mode in ('L1','DUAL')) else 'L0'
        mask_fy1 = _mask_by_mode_for_params(
            uav_idx=0, theta_deg=fy1_theta_deg, v=fy1_v,
            t_drop=fy1_t_drop, tau=fy1_tau,
            tgrid=tgrid, mode=eval_mode, PTS=PTS
        )
        union_fixed = np.logical_or(union_fixed, mask_fy1)
        r_drop = uav_pos(UAVS[0]["U0"], math.radians(fy1_theta_deg), fy1_v, fy1_t_drop)
        s_burst = burst_point(UAVS[0]["U0"], math.radians(fy1_theta_deg), fy1_v, fy1_t_drop, fy1_tau)
        fy1_row_for_report = {
            "uav": "FY1",
            "theta_deg": float(fy1_theta_deg),
            "v": float(fy1_v),
            "x_drop": float(r_drop[0]), "y_drop": float(r_drop[1]), "z_drop": float(r_drop[2]),
            "x_burst": float(s_burst[0]), "y_burst": float(s_burst[1]), "z_burst": float(s_burst[2]),
            "eff_time_s": float(mask_fy1.sum()*dt_mask),
            "mask": mask_fy1
        }
    elif fy1_intervals is not None:
        mask_fy1 = _intervals_to_mask(fy1_intervals, tgrid)
        union_fixed = np.logical_or(union_fixed, mask_fy1)
        fy1_row_for_report = {
            "uav": "FY1", "theta_deg": float('nan'), "v": float('nan'),
            "x_drop": float('nan'), "y_drop": float('nan'), "z_drop": float('nan'),
            "x_burst": float('nan'), "y_burst": float('nan'), "z_burst": float('nan'),
            "eff_time_s": float(mask_fy1.sum()*dt_mask),
            "mask": mask_fy1
        }
    else:
        raise ValueError("请至少提供 FY1 的参数 (theta/v/t_drop/tau) 或遮蔽区间 intervals")

    if mode in ('L0','L1'):
        cands, _ = build_candidates_q4(
            fracs=fracs, alphas=alphas, taus=taus,
            per_uav_keep=per_uav_keep, dt_mask=dt_mask,
            mask_mode=mode, PTS=PTS
        )
        if debug:
            counts_all = {i: sum(1 for cc in cands if cc["uav"]==i) for i in range(3)}
            counts_23 = {i: sum(1 for cc in cands if cc["uav"]==i) for i in (1,2)}
            print(f"[debug][fixed FY1] mask_mode={mode}, per-UAV nonzero candidates (all): {counts_all}")
            print(f"[debug][fixed FY1] using FY2/FY3: {counts_23}")
    else:
        cands_L0, _ = build_candidates_q4(
            fracs=fracs, alphas=alphas, taus=taus,
            per_uav_keep=per_uav_keep, dt_mask=dt_mask,
            mask_mode='L0', PTS=None
        )
        if debug:
            counts0 = {i: sum(1 for cc in cands_L0 if cc["uav"]==i) for i in range(3)}
            print("[debug][fixed FY1][DUAL] L0 pool per-UAV:", counts0)
        cands = remask_candidates(cands_L0, tgrid, 'L1', dt_mask, PTS=PTS)
        if dual_per_uav_keep_L1 is None:
            dual_per_uav_keep_L1 = per_uav_keep
        groups_tmp = {0:[],1:[],2:[]}
        for c in cands: groups_tmp[c["uav"]].append(c)
        cands_trim = []
        for u in [0,1,2]:
            groups_tmp[u].sort(key=lambda x:x["score"], reverse=True)
            cands_trim += groups_tmp[u][:dual_per_uav_keep_L1]
        cands = cands_trim
        if debug:
            counts1 = {i: sum(1 for cc in cands if cc["uav"]==i) for i in range(3)}
            print("[debug][fixed FY1][DUAL] L1 re-score pool per-UAV:", counts1)

    groups = {1:[], 2:[]}
    for c in cands:
        if c["uav"] in (1,2):
            groups[c["uav"]].append(c)

    if debug:
        print("[debug][fixed FY1] pool sizes (FY2/FY3):", {u:len(groups[u]) for u in (1,2)})

    (c2, c3), cover_total, union_mask = _pick_best_pair_given_fixed_union(groups, tgrid, union_fixed)

    ths = [fy1_row_for_report["theta_deg"], deg360(c2["theta"]), deg360(c3["theta"])]
    vs  = [fy1_row_for_report["v"], c2["v"], c3["v"]]
    tds = [fy1_t_drop if have_params else float('nan'), c2["t_drop"], c3["t_drop"]]
    taus_= [fy1_tau if have_params else float('nan'), c2["tau"], c3["tau"]]
    uidx = [0, 1, 2]
    unames = [UAVS[u]["name"] for u in uidx]

    bursts = []
    if have_params:
        sb1 = burst_point(UAVS[0]["U0"], math.radians(fy1_row_for_report["theta_deg"]), fy1_row_for_report["v"], fy1_t_drop, fy1_tau)
        bursts.append({
            "uav": "FY1",
            "t_drop": float(fy1_t_drop), "tau": float(fy1_tau),
            "t_burst": float(fy1_t_drop + fy1_tau),
            "s_burst": (float(sb1[0]), float(sb1[1]), float(sb1[2]))
        })
    for p in (c2, c3):
        u0 = UAVS[p["uav"]]["U0"]
        t_burst = p["t_drop"] + p["tau"]
        sb = burst_point(u0, p["theta"], p["v"], p["t_drop"], p["tau"])
        bursts.append({
            "uav": UAVS[p["uav"]]["name"],
            "t_drop": float(p["t_drop"]),
            "tau": float(p["tau"]),
            "t_burst": float(t_burst),
            "s_burst": (float(sb[0]), float(sb[1]), float(sb[2]))
        })
    bursts = sorted(bursts, key=lambda b: b["t_burst"])
    intervals = _mask_to_intervals(union_mask, tgrid)

    ans = {
        "method": "graph_max_coverage (FY1 fixed)",
        "strategy": ( "DUAL(L0→L1)" if mode=="DUAL" else mode ),
        "theta_deg": [round(x, 3) for x in ths],
        "v_u_mps":   [round(x, 3) for x in vs],
        "drops_s":   [round(x, 3) for x in tds],
        "taus_s":    [round(x, 3) for x in taus_],
        "uav_names": unames,
        "uav_indices": uidx,
        "bursts": bursts,
        "cover_total_s": float(cover_total),
        "cover_intervals_s": intervals,
        "config": {
            "dt_mask": dt_mask, "fracs": fracs, "alphas": alphas, "taus": taus,
            "per_uav_keep_base": per_uav_keep,
            "dual_per_uav_keep_L1": dual_per_uav_keep_L1,
            "N_ANG": N_ANG, "N_Z": N_Z, "INCLUDE_SIDE": INCLUDE_SIDE
        }
    }
    return ans

def convergence_test_dt_fixed(fy1_theta_deg, fy1_v, fy1_t_drop, fy1_tau,
                              strategy='DUAL', dts=(0.04, 0.02, 0.015, 0.01),
                              per_uav_keep=24, dual_per_uav_keep_L1=24,
                              N_ANG=48, N_Z=9, INCLUDE_SIDE=True):
    rows = []
    for dt in dts:
        ans = solve_q4_graph_fixed_fy1(
            strategy=strategy, dt_mask=dt,
            per_uav_keep=per_uav_keep, dual_per_uav_keep_L1=dual_per_uav_keep_L1,
            N_ANG=N_ANG, N_Z=N_Z, INCLUDE_SIDE=INCLUDE_SIDE, debug=False,
            fy1_theta_deg=fy1_theta_deg, fy1_v=fy1_v, fy1_t_drop=fy1_t_drop, fy1_tau=fy1_tau
        )
        rows.append((dt, float(ans["cover_total_s"])))
    return rows

def rotation_invariance_test_fixed(fy1_theta_deg, fy1_v, fy1_t_drop, fy1_tau,
                                   angles_deg=(0,45,90,135),
                                   strategy='DUAL', dt_mask=0.02,
                                   per_uav_keep=24, dual_per_uav_keep_L1=24,
                                   N_ANG=48, N_Z=9, INCLUDE_SIDE=True):
    def rotz(vec, ang_rad):
        c,s = math.cos(ang_rad), math.sin(ang_rad)
        x,y,z = vec[0], vec[1], vec[2]
        return np.array([c*x - s*y, s*x + c*y, z], dtype=float)

    M0_bak = M0.copy()
    U_bak = [u["U0"].copy() for u in UAVS]
    P_bak = P_TARGET.copy()
    C_bak = CYL_CENTER.copy()

    covs=[]
    for ang in angles_deg:
        rad = math.radians(ang)
        M0[:] = rotz(M0_bak, rad)
        for i in range(3):
            UAVS[i]["U0"][:] = rotz(U_bak[i], rad)
        P_TARGET[:] = rotz(P_bak, rad)
        CYL_CENTER[:] = rotz(C_bak, rad)
        ans = solve_q4_graph_fixed_fy1(
            strategy=strategy, dt_mask=dt_mask,
            per_uav_keep=per_uav_keep, dual_per_uav_keep_L1=dual_per_uav_keep_L1,
            N_ANG=N_ANG, N_Z=N_Z, INCLUDE_SIDE=INCLUDE_SIDE, debug=False,
            fy1_theta_deg=(fy1_theta_deg + ang),  # 增加旋转角
            fy1_v=fy1_v, fy1_t_drop=fy1_t_drop, fy1_tau=fy1_tau
        )
        covs.append(ans["cover_total_s"])

    M0[:] = M0_bak
    for i in range(3):
        UAVS[i]["U0"][:] = U_bak[i]
    P_TARGET[:] = P_bak
    CYL_CENTER[:] = C_bak

    return list(zip(angles_deg, covs))

def run_all_q4_checks_fixed(
    fy1_theta_deg=7.374506365477594,
    fy1_v=98.85283623880619,
    fy1_t_drop=0.025903266051691094,
    fy1_tau=0.8431654639833148,
    fy1_intervals=[(0.869068730035006, 5.929068730035007)]
):

    print("\n[check] 基本求解 + 一致性（FY1 固定，L0）")
    ans_L0_fix = solve_q4_graph_fixed_fy1(
        strategy='L0', dt_mask=0.015,
        per_uav_keep=28, debug=True,
        fy1_theta_deg=fy1_theta_deg, fy1_v=fy1_v, fy1_t_drop=fy1_t_drop, fy1_tau=fy1_tau,
        fy1_intervals=fy1_intervals
    )
    validate_q4_solution(ans_L0_fix, verbose=True)

    print("\n[check] DUAL 一致性（FY1 固定）")
    ans_DUAL_fix = solve_q4_graph_fixed_fy1(
        strategy='DUAL', dt_mask=0.02,
        per_uav_keep=28, dual_per_uav_keep_L1=24, debug=True,
        fy1_theta_deg=fy1_theta_deg, fy1_v=fy1_v, fy1_t_drop=fy1_t_drop, fy1_tau=fy1_tau,
        fy1_intervals=fy1_intervals
    )
    validate_q4_solution(ans_DUAL_fix, verbose=True)

    print("\n[check] 时间步长收敛（DUAL，FY1 固定）")
    rows = convergence_test_dt_fixed(
        fy1_theta_deg, fy1_v, fy1_t_drop, fy1_tau,
        strategy='DUAL', dts=(0.04, 0.02, 0.015, 0.01),
        per_uav_keep=24, dual_per_uav_keep_L1=24
    )
    print("  dt vs cover:", [(round(dt,3), round(cov,3)) for dt,cov in rows])

    print("\n[check] 旋转不变性（DUAL，FY1 固定）")
    rotrows = rotation_invariance_test_fixed(
        fy1_theta_deg, fy1_v, fy1_t_drop, fy1_tau,
        angles_deg=(0, 45, 90, 135), strategy='DUAL', dt_mask=0.02,
        per_uav_keep=24, dual_per_uav_keep_L1=24
    )
    print("  angle vs cover:", [(ang, round(cov,3)) for ang,cov in rotrows])

    return {
        "ans_L0_fix": ans_L0_fix,
        "ans_DUAL_fix": ans_DUAL_fix,
        "convergence_rows": rows,
        "rotation_rows": rotrows
    }


if __name__ == "__main__":
    ans_L0 = solve_q4_graph(
        strategy='L0',
        dt_mask=0.015,
        per_uav_keep=28,
        debug=True
    )
    print("\n[Q4 | Graph-L0] 最优：")
    for k, v in ans_L0.items():
        print(" ", k, ":", v)
    print("\n[Q4 | Graph-L0] 结果明细表：")
    print_q4_report_table(ans_L0)

    ans_L1 = solve_q4_graph(
        strategy='L1',
        dt_mask=0.02,
        N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
        per_uav_keep=24,
        debug=True
    )
    print("\n[Q4 | Graph-L1] 最优：")
    for k, v in ans_L1.items():
        print(" ", k, ":", v)
    print("\n[Q4 | Graph-L1] 结果明细表：")
    print_q4_report_table(ans_L1)

    ans_DUAL = solve_q4_graph(
        strategy='DUAL',
        dt_mask=0.02,
        per_uav_keep=28,
        dual_per_uav_keep_L1=24,
        N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
        debug=True
    )
    print("\n[Q4 | Graph-DUAL] 最优：")
    for k, v in ans_DUAL.items():
        print(" ", k, ":", v)
    print("\n[Q4 | Graph-DUAL] 结果明细表：")
    print_q4_report_table(ans_DUAL)

    ans_fixed = solve_q4_graph_fixed_fy1(
        strategy='DUAL',
        dt_mask=0.02,
        per_uav_keep=28,
        dual_per_uav_keep_L1=24,
        N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
        debug=True,
        fy1_theta_deg=7.374506365477594,
        fy1_v=98.85283623880619,
        fy1_t_drop=0.025903266051691094,
        fy1_tau=0.8431654639833148,
        fy1_intervals=[(0.869068730035006, 5.929068730035007)]
    )
    print("\n[Q4 | Graph-DUAL | FY1 fixed] 最优：")
    for k, v in ans_fixed.items():
        print(" ", k, ":", v)
    print("\n[Q4 | Graph-DUAL | FY1 fixed] 结果明细表：")
    print_q4_report_table(ans_fixed)


    #一致性校验
    print("\n[Q4 | 校验 | FY1 fixed]")
    validate_q4_solution(ans_fixed, verbose=True)

    _ = run_all_q4_checks_fixed(
        fy1_theta_deg=7.374506365477594,
        fy1_v=98.85283623880619,
        fy1_t_drop=0.025903266051691094,
        fy1_tau=0.8431654639833148,
        fy1_intervals=[(0.869068730035006, 5.929068730035007)]
    )
