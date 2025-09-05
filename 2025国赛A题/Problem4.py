# -*- coding: utf-8 -*-
"""
Q4（图论法，三策略 + “FY1 固定”降维版）
—— 仅针对“固定 FY1 策略”的模型检验：一致性、DUAL 一致性、步长收敛、旋转不变性

说明：
  1) 保留通用求解器 solve_q4_graph（L0/L1/DUAL），以及“FY1 固定版”求解器 solve_q4_graph_fixed_fy1
  2) 新增 run_all_q4_checks_fixed(...)：只对“固定 FY1”场景做四类检验
       - [check] 基本求解 + 一致性（L0 或 L1）
       - [check] DUAL 一致性（L0→L1 复评分）
       - [check] 时间步长收敛（DUAL）
       - [check] 旋转不变性（DUAL，旋转场景与 FY1 航向）
"""

import math
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from itertools import permutations

# =========================
# 一、常量与场景（按题面）
# =========================

g = 9.81                 # 重力加速度 (m/s^2)
VM = 300.0               # 导弹速度 (m/s)：指向假目标(原点)的匀速直线
V_SINK = 3.0             # 云团下沉速度 (m/s)
R_SMOKE = 10.0           # 云团有效半径 (m)
T_EFFECT = 20.0          # 起爆后有效时间 (s)

# 真目标：圆柱（半径7m、高10m），下底面圆心(0,200,0)
R_TAR, H_TAR = 7.0, 10.0
CYL_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)
P_TARGET = np.array([0.0, 200.0, 5.0], dtype=float)  # L0 代表点（圆柱几何中心）

# 导弹（以 M1 为例）
M0 = np.array([20000.0, 0.0, 2000.0], dtype=float)

# 三架无人机初始（FY1/FY2/FY3）
U0_FY1 = np.array([17800.0,     0.0, 1800.0], dtype=float)
U0_FY2 = np.array([12000.0,  1400.0, 1400.0], dtype=float)
U0_FY3 = np.array([ 6000.0, -3000.0,  700.0], dtype=float)
UAVS = [
    {"name": "FY1", "U0": U0_FY1},
    {"name": "FY2", "U0": U0_FY2},
    {"name": "FY3", "U0": U0_FY3},
]

def missile_hit_time(m0: np.ndarray) -> float:
    """导弹到达原点的时刻：T_hit = ||m0|| / VM（用于截断积分上限）"""
    return float(np.linalg.norm(m0) / VM)

T_HIT = missile_hit_time(M0)

# 角度显示归一化：[0,360)
def deg360(theta_rad: float) -> float:
    d = math.degrees(theta_rad)
    d = (d % 360.0 + 360.0) % 360.0
    return 0.0 if abs(d-360.0) < 1e-9 else d

# =========================
# 二、运动学与几何（L0/L1 判定所需）
# =========================

def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v

def missile_pos(m0: np.ndarray, t: float) -> np.ndarray:
    d = unit(-m0)
    return m0 + VM * d * t

def uav_pos(u0: np.ndarray, theta: float, v_u: float, t: float) -> np.ndarray:
    """无人机：等高度直线匀速；theta 为航向角（弧度）"""
    hx, hy = math.cos(theta), math.sin(theta)
    return np.array([u0[0] + v_u * hx * t, u0[1] + v_u * hy * t, u0[2]], dtype=float)

def burst_point(u0: np.ndarray, theta: float, v_u: float, t_drop: float, tau: float) -> np.ndarray:
    """起爆点（球心初值）= 投放点 + 水平惯性位移 + 自由落体位移"""
    hx, hy = math.cos(theta), math.sin(theta)
    r_drop = uav_pos(u0, theta, v_u, t_drop)
    horiz  = np.array([v_u * hx * tau, v_u * hy * tau, 0.0], dtype=float)
    vert   = np.array([0.0, 0.0, -0.5 * g * tau * tau], dtype=float)
    return r_drop + horiz + vert

def smoke_center_after_burst(s_burst: np.ndarray, t: float, t_burst: float) -> np.ndarray:
    """起爆后云团球心：以 3 m/s 匀速下沉"""
    dz = -V_SINK * max(0.0, t - t_burst)
    return s_burst + np.array([0.0, 0.0, dz], dtype=float)

def point_to_segment_dist(P: np.ndarray, Q: np.ndarray, X: np.ndarray) -> float:
    """点X到线段PQ的最小距离，用于球-线段相交判定"""
    v = Q - P
    vv = float(np.dot(v, v))
    if vv == 0.0:
        return float(np.linalg.norm(X - P))
    a = float(np.dot(X - P, v) / vv)
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    Y = P + a * v
    return float(np.linalg.norm(X - Y))

def covered_L0_at_time(m0, p_target, s_burst, t_burst, t) -> bool:
    """L0：以圆柱中心点为代表视轴，球-线段相交判定"""
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    return (point_to_segment_dist(p_target, m_t, s_t) <= R_SMOKE)

def clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

# ===== L1：圆柱采样与向量化判定 =====

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
    """构建圆柱表面的采样点集（上下圆面 + 可选侧壁）"""
    pts = [cyl_points_top_bottom(N_ang)]
    if include_side:
        pts.append(cyl_points_side(N_ang, N_z))
    return np.concatenate(pts, axis=0)

def covered_L1_at_time_vectorized(m0, s_burst, t_burst, t, PTS) -> bool:
    """L1：向量化判定任一点是否被遮蔽（对所有采样点 OR）"""
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    v = m_t - PTS            # (N,3)
    w = s_t - PTS            # (N,3)
    vv = np.sum(v * v, axis=1)
    alpha = np.divide(np.sum(w * v, axis=1), vv, out=np.zeros_like(vv), where=vv > 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    Y = PTS + alpha[:, None] * v
    dist = np.linalg.norm(s_t - Y, axis=1)
    return bool(np.any(dist <= R_SMOKE))

# =========================
# 三、图论：候选生成 + 掩码 + 贪心 + 兜底
# =========================

def _time_grid(dt: float = 0.02) -> np.ndarray:
    return np.arange(0.0, T_HIT + 1e-12, dt)

def _candidate_from_anchor(u0: np.ndarray, frac: float, alpha: float, tau_mult: float, clamp_eps: float = 0.10):
    """
    几何锚点生成单个候选（θ, v, t_drop, τ）
      - t_b = frac * min(60, T_HIT-2)，视线锚点 Y = P + alpha*(m(t_b)-P)
      - θ 指向 Y_xy，v 使 t_b 时刻到达 Y_xy（t_drop + τ ≈ t_b）
      - 竖直对齐：u0_z - 0.5 g τ^2 ≈ Y_z → τ_base
      - τ = tau_mult * τ_base，裁剪到 [0.2, min(12, t_b - clamp_eps)]
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
    """
    为每个 UAV 生成候选并计算遮蔽掩码（按 L0 或 L1）
    返回：候选列表 + 时间网格
    """
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
    """把候选重新用指定模式打分（L0→L1 用）"""
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

# ---------- 兜底：寻找最大未覆盖时间空档 ----------
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

# ---------- 选择：每个 UAV 恰好选 1 ----------
def select_exact_one_per_uav(
    candidates: List[Dict[str, Any]],
    tgrid: np.ndarray,
    mask_mode: str = 'L0',
    PTS: Optional[np.ndarray] = None
) -> Tuple[List[Dict[str, Any]], float, np.ndarray]:
    """分区拟阵 + 兜底；保证每个 UAV 选中 1 个候选"""
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

# ---------- 备用：至多 1 / UAV，最多 3 个 ----------
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

# =========================
# 四、统一求解接口：'L0' / 'L1' / 'DUAL'
# =========================

def solve_q4_graph(
    strategy: str = 'L0',          # 'L0' | 'L1' | 'DUAL'
    dt_mask: float = 0.015,
    # 候选池密度（可据算力调小）
    fracs=(0.12, 0.25, 0.40, 0.55, 0.70, 0.85, 0.92, 0.96, 0.985, 0.995),
    alphas=(0.60, 0.70, 0.80, 0.88, 0.92, 0.96, 0.985),
    taus=(0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50),
    per_uav_keep=28,
    # L1 采样（在 strategy='L1' 或 'DUAL' 的 L1 阶段生效）
    N_ANG: int = 48, N_Z: int = 9, INCLUDE_SIDE: bool = True,
    # 选择策略
    force_one_per_uav: bool = True,
    # DUAL 的 L1 阶段对候选再次筛选的每UAV上限（可适当更小以加速）
    dual_per_uav_keep_L1: Optional[int] = None,
    debug: bool = False
) -> Dict[str, Any]:

    mode = strategy.upper()
    if mode not in ('L0','L1','DUAL'):
        raise ValueError("strategy 需为 'L0' / 'L1' / 'DUAL'")

    # --- 准备 L1 采样（仅在需要 L1 的策略里构建一次）---#
    PTS = None
    if mode in ('L1','DUAL'):
        PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)

    # === A) 单策略：L0 或 L1 ===
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
            "uav_names": unames,             # ← 供重建/校验
            "uav_indices": uidx,             # ← 供重建/校验
            "bursts":  bursts,
            "cover_total_s": cover_val,
            "cover_intervals_s": intervals,
            "config": {
                "dt_mask": dt_mask, "fracs": fracs, "alphas": alphas, "taus": taus,
                "per_uav_keep": per_uav_keep, "force_one_per_uav": force_one_per_uav,
                **({"N_ANG": N_ANG, "N_Z": N_Z, "INCLUDE_SIDE": INCLUDE_SIDE} if mode!='L0' else {})
            }
        }

    # === B) 双策略：DUAL（L0→L1）===

    # 1) 用 L0 快速建池与初筛
    cands_L0, tgrid = build_candidates_q4(
        fracs=fracs, alphas=alphas, taus=taus,
        per_uav_keep=per_uav_keep, dt_mask=dt_mask,
        mask_mode='L0', PTS=None
    )
    if debug:
        counts0 = {i: sum(1 for c in cands_L0 if c["uav"]==i) for i in range(3)}
        print("[debug][DUAL] L0 pool per-UAV:", counts0)

    # 2) 对 L0 池“换评估模式”为 L1 并重新打分
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

    # 3) 在 L1 掩码下正式选择（含 gap 定制兜底）
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

# =========================
# 五、模型检验/复建通用工具
# =========================

def _reconstruct_union_from_ans(ans: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 ans 返回的参数重建并集掩码，作为一致性校验依据。
    依赖字段：
      - theta_deg, v_u_mps, drops_s, taus_s, uav_indices
      - config.dt_mask (+ 若 L1/DUAL 则还要 N_ANG/N_Z/INCLUDE_SIDE)
      - strategy
    """
    cfg = ans.get("config", {})
    dt = float(cfg.get("dt_mask", 0.015))
    tgrid = _time_grid(dt)
    strategy = str(ans.get("strategy", "L0")).upper()
    mode = 'L1' if ('L1' in strategy) else 'L0'

    # 构建 PTS（若 L1）
    PTS = None
    if mode == 'L1':
        PTS = build_cylinder_samples(
            N_ang=int(cfg.get("N_ANG", 48)),
            N_z=int(cfg.get("N_Z", 9)),
            include_side=bool(cfg.get("INCLUDE_SIDE", True))
        )

    # 取数组（保持顺序一一对应）
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
    """
    校验：
      1) 由返回参数重建的并集覆盖时长与 ans['cover_total_s'] 接近
      2) 由掩码恢复的区间与 ans['cover_intervals_s'] 大体一致（离散步长允许±dt）
    """
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
    # 宽松比较：区间个数一致，端点差在 2*dt 内
    assert len(inter) == len(inter_ans), "区间数量不一致"
    for (a,b), (A,B) in zip(inter, inter_ans):
        assert abs(a-A) <= 2.5*dt and abs(b-B) <= 2.5*dt, "区间端点不一致（离散误差之外）"

# =========================
# 六、Q4 结果“明细表”（可选）
# =========================

def _mode_from_strategy_str(strategy_str: str) -> str:
    """从返回的 strategy 文本判断掩码评估模式：包含 'L1' 则用 L1，否则用 L0。"""
    s = (strategy_str or "").upper()
    return 'L1' if ('L1' in s) else 'L0'

def summarize_solution_rows(ans: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    输入：solve_q4_graph(...) 或 solve_q4_graph_fixed_fy1(...) 的返回字典 ans
    输出：每架 UAV 对应的一行信息（用于打印或导出）
    """
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
    """中文列名明细表"""
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

# =========================
# 七、FY1 已知最优 → 两机优化（降维版）
# =========================

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
    groups,               # {uav_idx: [candidates...]} 仅含剩余两机
    tgrid,
    union_fixed           # FY1 的固定掩码
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
    strategy: str = 'DUAL',          # 'L0' | 'L1' | 'DUAL'
    dt_mask: float = 0.02,
    fracs=(0.12, 0.25, 0.40, 0.55, 0.70, 0.85, 0.92, 0.96, 0.985, 0.995),
    alphas=(0.60, 0.70, 0.80, 0.88, 0.92, 0.96, 0.985),
    taus=(0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50),
    per_uav_keep=28,
    dual_per_uav_keep_L1: Optional[int] = None,
    N_ANG: int = 48, N_Z: int = 9, INCLUDE_SIDE: bool = True,
    debug: bool = False,
    # —— FY1 固定解（建议用两阶段的参数；也可只给区间）
    fy1_theta_deg: Optional[float] = None,
    fy1_v: Optional[float] = None,
    fy1_t_drop: Optional[float] = None,
    fy1_tau: Optional[float] = None,
    fy1_intervals: Optional[List[Tuple[float, float]]] = None
) -> Dict[str, Any]:
    """
    固定 FY1 的遮蔽（由参数或区间给出），只优化 FY2、FY3 的组合以最大化并集时长。
    支持 L0/L1/DUAL 三策略。
    """
    mode = strategy.upper()
    if mode not in ('L0','L1','DUAL'):
        raise ValueError("strategy 需为 'L0' / 'L1' / 'DUAL'")

    # 时间网格与（必要时）L1 采样
    tgrid = _time_grid(dt_mask)
    PTS = None
    if mode in ('L1','DUAL'):
        PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)

    # ===== 1) 构造 FY1 的固定掩码 union_fixed =====
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

    # ===== 2) 为 FY2/FY3 生成候选池（按策略）并重评分 =====
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
    else:  # DUAL：L0 建池 → L1 重打分
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

    # 仅保留 FY2/FY3
    groups = {1:[], 2:[]}
    for c in cands:
        if c["uav"] in (1,2):
            groups[c["uav"]].append(c)

    if debug:
        print("[debug][fixed FY1] pool sizes (FY2/FY3):", {u:len(groups[u]) for u in (1,2)})

    # ===== 3) 在 FY1 掩码已固定条件下，枚举 FY2×FY3 取最优 =====
    (c2, c3), cover_total, union_mask = _pick_best_pair_given_fixed_union(groups, tgrid, union_fixed)

    # ===== 4) 汇总答案（顺序按 FY1, FY2, FY3）=====
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

# =========================
# 八、仅“固定 FY1 策略”的模型检验
# =========================

def convergence_test_dt_fixed(fy1_theta_deg, fy1_v, fy1_t_drop, fy1_tau,
                              strategy='DUAL', dts=(0.04, 0.02, 0.015, 0.01),
                              per_uav_keep=24, dual_per_uav_keep_L1=24,
                              N_ANG=48, N_Z=9, INCLUDE_SIDE=True):
    """步长收敛（固定 FY1）"""
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
    """旋转不变性（固定 FY1）：旋转场景，同时 FY1 航向角 +angle"""
    def rotz(vec, ang_rad):
        c,s = math.cos(ang_rad), math.sin(ang_rad)
        x,y,z = vec[0], vec[1], vec[2]
        return np.array([c*x - s*y, s*x + c*y, z], dtype=float)

    # 备份原始几何
    M0_bak = M0.copy()
    U_bak = [u["U0"].copy() for u in UAVS]
    P_bak = P_TARGET.copy()
    C_bak = CYL_CENTER.copy()

    covs=[]
    for ang in angles_deg:
        rad = math.radians(ang)
        # 旋转场景
        M0[:] = rotz(M0_bak, rad)
        for i in range(3):
            UAVS[i]["U0"][:] = rotz(U_bak[i], rad)
        P_TARGET[:] = rotz(P_bak, rad)
        CYL_CENTER[:] = rotz(C_bak, rad)
        # FY1 航向角同步旋转
        ans = solve_q4_graph_fixed_fy1(
            strategy=strategy, dt_mask=dt_mask,
            per_uav_keep=per_uav_keep, dual_per_uav_keep_L1=dual_per_uav_keep_L1,
            N_ANG=N_ANG, N_Z=N_Z, INCLUDE_SIDE=INCLUDE_SIDE, debug=False,
            fy1_theta_deg=(fy1_theta_deg + ang),  # 增加旋转角
            fy1_v=fy1_v, fy1_t_drop=fy1_t_drop, fy1_tau=fy1_tau
        )
        covs.append(ans["cover_total_s"])

    # 复原
    M0[:] = M0_bak
    for i in range(3):
        UAVS[i]["U0"][:] = U_bak[i]
    P_TARGET[:] = P_bak
    CYL_CENTER[:] = C_bak

    return list(zip(angles_deg, covs))

def run_all_q4_checks_fixed(
    # FY1 固定参数（来自你的 two-stage 结果）
    fy1_theta_deg=7.374506365477594,
    fy1_v=98.85283623880619,
    fy1_t_drop=0.025903266051691094,
    fy1_tau=0.8431654639833148,
    fy1_intervals=[(0.869068730035006, 5.929068730035007)]
):
    """
    仅针对“固定 FY1”的四类检验：
      - 基本求解 + 一致性（L0）
      - DUAL 一致性
      - 时间步长收敛（DUAL）
      - 旋转不变性（DUAL）
    """
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

# =========================
# 主程序（示例：仅跑“固定 FY1”检验）
# =========================
if __name__ == "__main__":
    # —— L0：最快
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

    # —— L1：高保真（计算量更大，建议适度增大 dt_mask 或减少候选）
    ans_L1 = solve_q4_graph(
        strategy='L1',
        dt_mask=0.02,        # L1 建议 0.02~0.03
        N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
        per_uav_keep=24,     # L1 可适当减小以控时
        debug=True
    )
    print("\n[Q4 | Graph-L1] 最优：")
    for k, v in ans_L1.items():
        print(" ", k, ":", v)
    print("\n[Q4 | Graph-L1] 结果明细表：")
    print_q4_report_table(ans_L1)

    # —— DUAL：先 L0 建池，后 L1 重打分重选（快 & 稳）
    ans_DUAL = solve_q4_graph(
        strategy='DUAL',
        dt_mask=0.02,               # DUAL 的 L1 阶段也用到该步长
        per_uav_keep=28,
        dual_per_uav_keep_L1=24,    # L1 阶段每UAV保留数量（可再小一点以提速）
        N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
        debug=True
    )
    print("\n[Q4 | Graph-DUAL] 最优：")
    for k, v in ans_DUAL.items():
        print(" ", k, ":", v)
    print("\n[Q4 | Graph-DUAL] 结果明细表：")
    print_q4_report_table(ans_DUAL)

    # —— FY1 固定：用你 two-stage 的 FY1 最优参数，优化 FY2/FY3（DUAL，最终 L1 评估）
    ans_fixed = solve_q4_graph_fixed_fy1(
        strategy='DUAL',
        dt_mask=0.02,
        per_uav_keep=28,
        dual_per_uav_keep_L1=24,
        N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
        debug=True,
        # FY1 two-stage 最优参数（你提供的数据）
        fy1_theta_deg=7.374506365477594,
        fy1_v=98.85283623880619,
        fy1_t_drop=0.025903266051691094,
        fy1_tau=0.8431654639833148,
        # 同时给出 FY1 的遮蔽区间（优化不依赖，但便于参考）
        fy1_intervals=[(0.869068730035006, 5.929068730035007)]
    )
    print("\n[Q4 | Graph-DUAL | FY1 fixed] 最优：")
    for k, v in ans_fixed.items():
        print(" ", k, ":", v)
    print("\n[Q4 | Graph-DUAL | FY1 fixed] 结果明细表：")
    print_q4_report_table(ans_fixed)

    # —— 一致性校验（可选）
    print("\n[Q4 | 校验 | FY1 fixed]")
    validate_q4_solution(ans_fixed, verbose=True)
    _ = run_all_q4_checks_fixed(
        fy1_theta_deg=7.374506365477594,
        fy1_v=98.85283623880619,
        fy1_t_drop=0.025903266051691094,
        fy1_tau=0.8431654639833148,
        fy1_intervals=[(0.869068730035006, 5.929068730035007)]
    )
