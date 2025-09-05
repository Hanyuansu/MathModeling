# -*- coding: utf-8 -*-
"""
Q4（图论法，纯 L0 侧视）—— 三架 UAV 各投一枚，最大化遮蔽“并集时长”
方法：
  1) 候选生成（几何锚点）：为每架 UAV 构造若干 (θ, v, t_drop, τ)
  2) L0 快速评估：把每个候选映射为时间网格上的布尔掩码
  3) 分区拟阵子模贪心 + “最大空档定制”兜底：每架 UAV 选 1 个候选
  4) 从并集掩码恢复遮蔽区间，输出整洁结果
"""

import math
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

# =========================
# 二、运动学与几何（L0 判定所需）
# =========================

def unit(v: np.ndarray) -> np.ndarray:
    """单位向量；零向量保持零"""
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v

def missile_pos(m0: np.ndarray, t: float) -> np.ndarray:
    """导弹轨迹：m(t) = m0 + VM * d * t，d 指向原点"""
    d = unit(-m0)
    return m0 + VM * d * t

def uav_pos(u0: np.ndarray, theta: float, v_u: float, t: float) -> np.ndarray:
    """无人机：等高度直线匀速；theta 为航向角（弧度）"""
    hx, hy = math.cos(theta), math.sin(theta)
    return np.array([u0[0] + v_u * hx * t, u0[1] + v_u * hy * t, u0[2]], dtype=float)

def burst_point(u0: np.ndarray, theta: float, v_u: float, t_drop: float, tau: float) -> np.ndarray:
    """
    起爆点（球心初值）= 投放点 + 水平惯性位移 + 自由落体位移
      r_drop = uav_pos(u0, theta, v_u, t_drop)
      s_burst = r_drop + [v_u*hx*tau, v_u*hy*tau, -0.5*g*tau^2]
    """
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
    """点 X 到线段 PQ 的最小距离"""
    v = Q - P
    vv = float(np.dot(v, v))
    if vv == 0.0:
        return float(np.linalg.norm(X - P))
    a = float(np.dot(X - P, v) / vv)
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    Y = P + a * v
    return float(np.linalg.norm(X - Y))

def covered_L0_at_time(m0, p_target, s_burst, t_burst, t) -> bool:
    """L0 单时刻判定：球（云团）与“目标代表点→导弹”的视线段是否相交"""
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    return (point_to_segment_dist(p_target, m_t, s_t) <= R_SMOKE)

def clip(x, lo, hi):
    """数值裁剪到 [lo, hi]"""
    return lo if x < lo else (hi if x > hi else x)

# =========================
# 三、图论：候选生成 + 掩码 + 贪心 + 兜底
# =========================

def _time_grid(dt: float = 0.02) -> np.ndarray:
    """全局时间网格 [0, T_HIT]（离散化后用于最大覆盖）"""
    return np.arange(0.0, T_HIT + 1e-12, dt)

def _candidate_from_anchor(u0: np.ndarray, frac: float, alpha: float, tau_mult: float, clamp_eps: float = 0.10):
    """
    几何锚点生成单个候选（θ, v, t_drop, τ）
      - t_b = frac * min(60, T_HIT-2)，视线锚点 Y = P + alpha*(m(t_b)-P)
      - θ 指向 Y_xy，v 使 t_b 时刻到达 Y_xy（t_drop + τ ≈ t_b）
      - 竖直对齐：u0_z - 0.5 g τ^2 ≈ Y_z → τ_base
      - τ = tau_mult * τ_base，裁剪到 [0.2, min(12, t_b - clamp_eps)] 确保 t_drop≥clamp_eps
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
    """计算候选在时间网格上的遮蔽掩码（L0）"""
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

def build_candidates_q4(
    fracs = (0.12, 0.25, 0.40, 0.55, 0.70, 0.85, 0.92, 0.96),
    alphas = (0.60, 0.70, 0.80, 0.88, 0.92, 0.96, 0.985),
    taus   = (0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50),
    per_uav_keep: int = 28,
    dt_mask: float = 0.015
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """为每个 UAV 生成若干候选并计算 L0 掩码与得分（遮蔽时长）"""
    tgrid = _time_grid(dt_mask)
    all_cands = []
    for uav_idx in range(3):
        local = []
        u0 = UAVS[uav_idx]["U0"]
        for f in fracs:
            for a in alphas:
                for tau_mult in taus:
                    th, v, td, ta = _candidate_from_anchor(u0, f, a, tau_mult)
                    mask = _candidate_mask_L0(uav_idx, th, v, td, ta, tgrid)
                    score = float(mask.sum() * dt_mask)
                    # 只保留有贡献的候选；若某 UAV 全为 0，选择阶段有兜底
                    if score > 0.0:
                        local.append({
                            "uav": uav_idx,
                            "theta": th, "v": v, "t_drop": td, "tau": ta,
                            "mask": mask, "score": score
                        })
        local.sort(key=lambda x: x["score"], reverse=True)
        all_cands += local[:per_uav_keep]
    return all_cands, tgrid

# ---------- 兜底：寻找最大未覆盖时间空档 ----------
def _largest_gap(mask: np.ndarray, tgrid: np.ndarray) -> Tuple[float, float]:
    """返回最大未覆盖区间的 (center_t, length)；若无空档返回 (中点, 0)"""
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
                                   tgrid: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    针对某 UAV 在 gap_center_t 附近合成一个候选，并做一个很小的坐标下降抛光（纯 L0）
    返回字典（含 mask/score）；若失败返回 None
    """
    u0 = UAVS[uav_idx]["U0"]
    # 用 gap 中心做时间锚，偏向导弹端 alpha
    t_b = clip(gap_center_t, 0.6, min(59.5, T_HIT-0.5))
    frac = t_b / min(60.0, T_HIT-2.0)
    alphas_try = (0.92, 0.96, 0.985)
    tau_mult_try = (0.70, 0.85, 1.00, 1.15)

    def mk(theta, v, td, ta):
        mask = _candidate_mask_L0(uav_idx, theta, v, td, ta, tgrid)
        return {"uav": uav_idx, "theta":theta, "v":v, "t_drop":td, "tau":ta,
                "mask":mask, "score": float(mask.sum()*(tgrid[1]-tgrid[0]))}

    best = None
    for a in alphas_try:
        th, v, td, ta = _candidate_from_anchor(u0, frac, a, 1.0)
        # 小邻域抛光
        neigh_yaw = [th + math.radians(d) for d in (-8,-4,0,4,8)]
        neigh_v   = [clip(v + dv, 70.0, 140.0) for dv in (-12,-6,0,6,12)]
        neigh_td  = [clip(td + d, 0.0, 60.0) for d in (-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5)]
        neigh_tau = [clip(ta*m, 0.2, min(12.0, t_b-0.1)) for m in tau_mult_try]
        for th1 in neigh_yaw:
            for v1 in neigh_v:
                for td1 in neigh_td:
                    for ta1 in neigh_tau:
                        c1 = mk(th1, v1, td1, ta1)
                        if best is None or c1["score"] > best["score"]:
                            best = c1
    return best

# ---------- 选择：每个 UAV 恰好选 1 ----------
def select_exact_one_per_uav(
    candidates: List[Dict[str, Any]],
    tgrid: np.ndarray
) -> Tuple[List[Dict[str, Any]], float, np.ndarray]:
    """
    强制约束：每个 UAV 恰好选 1 个候选。
    做法：枚举3个UAV的选择顺序（3!），对每个顺序依次在当前并集上挑该UAV贡献最大的候选；
          若该 UAV 所有候选边际增益≤0，则调用“最大空档定制 + 小抛光”生成一个新候选。
    """
    # 分组
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
            # 兜底：该 UAV 没有正增益 → 在最大空档附近合成+抛光一个
            if best_c is None or best_gain <= 0.0:
                center_t, _ = _largest_gap(union_mask, tgrid)
                synth = _synthesize_and_polish_for_gap(u, center_t, tgrid)
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

# ---------- 备用：分区拟阵子模贪心（至多 1 / UAV，最多 3 个） ----------
def greedy_partition_matroid_max_coverage(
    candidates: List[Dict[str, Any]],
    tgrid: np.ndarray,
    per_uav_limit: int = 1,
    K_total: int = 3
) -> Tuple[List[Dict[str, Any]], float, np.ndarray]:
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
    """把并集掩码转回连续时间区间列表"""
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
# 四、统一求解接口（纯 L0）
# =========================

def solve_q4_graph_L0(
    dt_mask: float = 0.015,
    fracs=(0.12, 0.25, 0.40, 0.55, 0.70, 0.85, 0.92, 0.96, 0.985, 0.995),
    alphas=(0.60, 0.70, 0.80, 0.88, 0.92, 0.96, 0.985),
    taus=(0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50),
    per_uav_keep=28,
    force_one_per_uav: bool = True,
    debug: bool = False
) -> Dict[str, Any]:
    # 1) 候选
    cands, tgrid = build_candidates_q4(
        fracs=fracs, alphas=alphas, taus=taus, per_uav_keep=per_uav_keep, dt_mask=dt_mask
    )
    if debug:
        counts = {i: sum(1 for c in cands if c["uav"]==i) for i in range(3)}
        print("[debug] per-UAV nonzero candidates:", counts)

    # 2) 选择（默认：每 UAV 必选 1）
    if force_one_per_uav:
        picked, cover_L0, union_mask = select_exact_one_per_uav(cands, tgrid)
    else:
        picked, cover_L0, union_mask = greedy_partition_matroid_max_coverage(
            candidates=cands, tgrid=tgrid, per_uav_limit=1, K_total=3
        )

    # 3) 整理输出
    ths = [p["theta"] for p in picked]
    vs  = [p["v"] for p in picked]
    tds = [p["t_drop"] for p in picked]
    taus_= [p["tau"] for p in picked]

    # 起爆信息（按时序）
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
        "stage": "L0_greedy_only + gap_synthesis",
        "theta_deg": [round(math.degrees(x), 3) for x in ths],
        "v_u_mps": [round(x, 3) for x in vs],
        "drops_s": [round(x, 3) for x in tds],
        "taus_s":  [round(x, 3) for x in taus_],
        "bursts":  bursts,
        "cover_total_s": cover_L0,
        "cover_intervals_s": intervals,
        "config": {"dt_mask": dt_mask, "fracs": fracs, "alphas": alphas, "taus": taus,
                   "per_uav_keep": per_uav_keep, "force_one_per_uav": force_one_per_uav}
    }

# =========================
# 主程序
# =========================
if __name__ == "__main__":
    ans = solve_q4_graph_L0(
        dt_mask=0.015,
        fracs=(0.12, 0.25, 0.40, 0.55, 0.70, 0.85, 0.92, 0.96, 0.985, 0.995),
        alphas=(0.60, 0.70, 0.80, 0.88, 0.92, 0.96, 0.985),
        taus=(0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50),
        per_uav_keep=28,
        force_one_per_uav=True,
        debug=True
    )
    print("[Q4 | Graph-L0] 最优：")
    for k, v in ans.items():
        print(" ", k, ":", v)
