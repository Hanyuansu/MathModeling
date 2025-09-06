# -*- coding: utf-8 -*-
"""
Q5：五架 UAV、每架 ≤3 枚，对三枚导弹的联合遮蔽（先 L0 再 L1），用满 15 枚

要点：
- 【锁定规则】每架 UAV 在“第一枚投放”时锁定 (theta, v)，之后该机所有投放必须同航向同速度（容差可设为 0）。
- 【间隔约束】全流程统一以“投放时刻 t_drop”为基准，保持同一架 UAV 的相邻投放满足 t_drop 间隔 ≥ MIN_DROP_GAP（默认 1s）。
- 两阶段：L0 生成候选+初始解 → L1 小抛光（仅调 t_drop/tau，不改 theta/v），抛光阶段也执行 t_drop 间隔检查。
- 时间步长统一：DT_STEP（默认 0.015），L0/L1 使用同一 dt，避免统计口径不一致。
- 先确保“三枚导弹均被至少一枚覆盖”，再继续贪心加弹，最后若仍不足 15 枚，用“零遮蔽填充弹”补满。
- 报表 12 列；零遮蔽弹也输出一行（导弹编号为“-”，有效时长=0.0）。

作者：ChatGPT
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# =========================
# 场景与常量
# =========================
g = 9.81
VM = 300.0              # 导弹速度
V_SINK = 3.0            # 烟团下沉速度
R_SMOKE = 10.0          # 等效半径
T_EFFECT = 20.0         # 单次起爆有效时长

# 统一时间步长（L0/L1 同步）
DT_STEP = 0.015

# 目标：圆柱
R_TAR, H_TAR = 7.0, 10.0
CYL_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)
P_TARGET   = np.array([0.0, 200.0, 5.0], dtype=float)

# 三枚导弹（给定初值）
MISSILES = [
    {"name": "M1", "M0": np.array([20000.0,    0.0, 2000.0], dtype=float)},
    {"name": "M2", "M0": np.array([19000.0,  600.0, 2100.0], dtype=float)},
    {"name": "M3", "M0": np.array([18000.0, -600.0, 1900.0], dtype=float)},
]

# 五架 UAV（初始位置、高度）
UAVS = [
    {"name": "FY1", "U0": np.array([17800.0,     0.0, 1800.0], dtype=float)},
    {"name": "FY2", "U0": np.array([12000.0,  1400.0, 1400.0], dtype=float)},
    {"name": "FY3", "U0": np.array([ 6000.0, -3000.0,  700.0], dtype=float)},
    {"name": "FY4", "U0": np.array([11000.0,  2000.0, 1800.0], dtype=float)},
    {"name": "FY5", "U0": np.array([13000.0, -2000.0, 1300.0], dtype=float)},
]

# 每架最多 3 枚
BUDGETS = (3, 3, 3, 3, 3)
MIN_DROP_GAP = 1.0      # 【规则】同机投放时刻 t_drop 最小间隔（秒）

# 航向/速度锁定容差（可设严一些确保完全锁定）
HEADING_TOL_DEG = 2.0
SPEED_TOL = 1.5

# 采样参数（候选生成锚点）
FRACS  = (0.10, 0.18, 0.25, 0.40, 0.55, 0.70, 0.85, 0.92, 0.96, 0.985)
ALPHAS = (0.60, 0.70, 0.80, 0.88, 0.92, 0.96, 0.985)
TAUS_MUL = (0.55, 0.70, 0.85, 1.00, 1.15, 1.30)
PER_UAV_KEEP = 36
DEDUP_EPS = 0.12

# L1 目标采样
N_ANG, N_Z, INCLUDE_SIDE = 48, 9, True
POLISH_ROUNDS = 1        # L1 坐标下降轮次（适度即可）


# =========================
# 基本几何/运动学
# =========================
def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v

def missile_hit_time(m0: np.ndarray) -> float:
    return float(np.linalg.norm(m0) / VM)

def missile_pos(m0: np.ndarray, t: float) -> np.ndarray:
    d = unit(-m0)
    return m0 + VM * d * t

def uav_pos(u0: np.ndarray, theta: float, v_u: float, t: float) -> np.ndarray:
    hx, hy = math.cos(theta), math.sin(theta)
    return np.array([u0[0] + v_u*hx*t, u0[1] + v_u*hy*t, u0[2]], dtype=float)

def burst_point(u0: np.ndarray, theta: float, v_u: float, t_drop: float, tau: float) -> np.ndarray:
    hx, hy = math.cos(theta), math.sin(theta)
    r_drop = uav_pos(u0, theta, v_u, t_drop)
    horiz  = np.array([v_u*hx*tau, v_u*hy*tau, 0.0], dtype=float)
    vert   = np.array([0.0, 0.0, -0.5*g*tau*tau], dtype=float)
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
    a = 0.0 if a < 0 else (1.0 if a > 1.0 else a)
    Y = P + a * v
    return float(np.linalg.norm(X - Y))

def clip(x, lo, hi): return lo if x < lo else (hi if x > hi else x)

# L0 判定：目标中心点代表视轴
def covered_L0_at_time(m0, p_target, s_burst, t_burst, t) -> bool:
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    return (point_to_segment_dist(p_target, m_t, s_t) <= R_SMOKE)

# L1 采样
def cyl_points_top_bottom(N_ang: int = 48) -> np.ndarray:
    cx, cy, _ = CYL_CENTER; out=[]
    for z in (0.0, H_TAR):
        for k in range(N_ang):
            ang = 2.0*math.pi*k/N_ang
            out.append((cx+R_TAR*math.cos(ang), cy+R_TAR*math.sin(ang), z))
    return np.array(out, dtype=float)

def cyl_points_side(N_ang: int = 48, N_z: int = 9) -> np.ndarray:
    cx, cy, _ = CYL_CENTER; zs = np.linspace(0.0, H_TAR, N_z); out=[]
    for z in zs:
        for k in range(N_ang):
            ang = 2.0*math.pi*k/N_ang
            out.append((cx+R_TAR*math.cos(ang), cy+R_TAR*math.sin(ang), z))
    return np.array(out, dtype=float)

def build_cylinder_samples(N_ang=48, N_Z=9, include_side=True) -> np.ndarray:
    pts=[cyl_points_top_bottom(N_ang)]
    if include_side: pts.append(cyl_points_side(N_ang, N_Z))
    return np.concatenate(pts, axis=0)

def covered_L1_at_time_vectorized(m0: np.ndarray, s_burst: np.ndarray, t_burst: float, t: float, PTS: np.ndarray) -> bool:
    m_t = missile_pos(m0, t);  s_t = smoke_center_after_burst(s_burst, t, t_burst)
    v = m_t - PTS;  w = s_t - PTS
    vv = np.sum(v*v, axis=1)
    alpha = np.divide(np.sum(w*v, axis=1), vv, out=np.zeros_like(vv), where=vv>0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    Y = PTS + alpha[:, None]*v
    dist = np.linalg.norm(s_t - Y, axis=1)
    return bool(np.any(dist <= R_SMOKE))


# =========================
# 掩码/时间网格与打分（只看覆盖）
# =========================
def _time_grid(T_hit: float, dt: float) -> np.ndarray:
    return np.arange(0.0, T_hit + 1e-12, dt)

def _mask_for_candidate_L0(uav_idx: int, cand, tgrids: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    masks={}
    u0 = UAVS[uav_idx]["U0"]
    th, v, td, ta = cand["theta"], cand["v"], cand["t_drop"], cand["tau"]
    t_burst = td + ta
    for m in MISSILES:
        name, m0 = m["name"], m["M0"]
        T_HIT = missile_hit_time(m0)
        tgrid = tgrids[name]
        mask = np.zeros_like(tgrid, dtype=bool)
        if t_burst < T_HIT:
            sb = burst_point(u0, th, v, td, ta)
            if sb[2] > 0.0:
                t_start, t_end = t_burst, min(t_burst + T_EFFECT, T_HIT)
                idx = np.where((tgrid >= t_start) & (tgrid <= t_end))[0]
                for k in idx:
                    t = float(tgrid[k])
                    if covered_L0_at_time(m0, P_TARGET, sb, t_burst, t):
                        mask[k] = True
        masks[name]=mask
    return masks

def _mask_for_candidate_L1(uav_idx: int, cand, tgrids: Dict[str, np.ndarray], PTS: np.ndarray) -> Dict[str, np.ndarray]:
    masks={}
    u0 = UAVS[uav_idx]["U0"]
    th, v, td, ta = cand["theta"], cand["v"], cand["t_drop"], cand["tau"]
    t_burst = td + ta
    for m in MISSILES:
        name, m0 = m["name"], m["M0"]
        T_HIT = missile_hit_time(m0)
        tgrid = tgrids[name]
        mask = np.zeros_like(tgrid, dtype=bool)
        if t_burst < T_HIT:
            sb = burst_point(u0, th, v, td, ta)
            if sb[2] > 0.0:
                t_start, t_end = t_burst, min(t_burst + T_EFFECT, T_HIT)
                idx = np.where((tgrid >= t_start) & (tgrid <= t_end))[0]
                for k in idx:
                    t = float(tgrid[k])
                    if covered_L1_at_time_vectorized(m0, sb, t_burst, t, PTS):
                        mask[k] = True
        masks[name]=mask
    return masks

def _cover_sum(masks: Dict[str, np.ndarray], dt: float) -> float:
    return float(sum(mask.sum() for mask in masks.values()) * dt)


# =========================
# 候选生成（以导弹锚点反推）
# =========================
def _candidate_from_anchor(u0: np.ndarray, m0: np.ndarray,
                           frac: float, alpha: float, tau_mult: float,
                           clamp_eps: float = 0.10):
    T_HIT = missile_hit_time(m0)
    t_b = clip(frac * min(60.0, T_HIT-2.0), 0.5, 59.5)
    m_tb = missile_pos(m0, t_b)
    Y = P_TARGET + alpha*(m_tb - P_TARGET)  # 预瞄起爆点

    dx, dy = Y[0]-u0[0], Y[1]-u0[1]
    theta = math.atan2(dy, dx)
    D_xy  = math.hypot(dx, dy)
    v = clip(D_xy / t_b, 70.0, 140.0)

    u0z, Yz = float(u0[2]), float(Y[2])
    tau_base = math.sqrt(max(0.0, 2.0*(u0z - Yz))/g) if u0z > Yz else 0.2
    tau_max_by_t = max(0.2, t_b - clamp_eps)
    tau = clip(tau_base * tau_mult, 0.2, min(12.0, tau_max_by_t))
    t_drop = t_b - tau
    return theta%(2.0*math.pi), v, t_drop, tau

def build_candidates_L0(
    dt: float = DT_STEP,
    fracs=FRACS, alphas=ALPHAS, taus=TAUS_MUL,
    per_uav_keep: int = PER_UAV_KEEP, dedup_eps: float = DEDUP_EPS
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    tgrids = {m["name"]: _time_grid(missile_hit_time(m["M0"]), dt) for m in MISSILES}
    all_cands=[]
    for u in range(len(UAVS)):
        u0 = UAVS[u]["U0"]; local=[]
        for m in MISSILES:
            m0 = m["M0"]
            for f in fracs:
                for a in alphas:
                    for mul in taus:
                        th, v, td, ta = _candidate_from_anchor(u0, m0, f, a, mul)
                        cand = {"uav": u, "theta": th, "v": v, "t_drop": td, "tau": ta}
                        masks = _mask_for_candidate_L0(u, cand, tgrids)
                        score = _cover_sum(masks, dt)
                        if score <= 0.0:  # 仅保留有效候选；零遮蔽稍后用“填充弹”机制处理
                            continue
                        cand.update({"mask_by_missile": masks, "score_cover": score, "t_burst": td+ta})
                        local.append(cand)
        # 轻度按 t_burst 去重
        local.sort(key=lambda c:c["t_burst"])
        filtered=[]
        for c in local:
            if not filtered or abs(c["t_burst"]-filtered[-1]["t_burst"])>dedup_eps:
                filtered.append(c)
            else:
                if c["score_cover"] > filtered[-1]["score_cover"]:
                    filtered[-1]=c
        filtered.sort(key=lambda c:c["score_cover"], reverse=True)
        all_cands += filtered[:per_uav_keep]
    return all_cands, tgrids


# =========================
# 约束与增益
# =========================
def _angdiff(a: float, b: float) -> float:
    d = abs((a - b + math.pi) % (2*math.pi) - math.pi)
    return d

def _feasible_with(cand, chosen, budgets, min_drop_gap,
                   locks: Dict[int, Optional[Tuple[float,float]]],
                   heading_tol_rad: float, speed_tol: float) -> bool:
    u = cand["uav"]
    if budgets[u] <= 0: return False
    # ——【关键】用投放时刻 t_drop 做 1s 间隔限制——
    td_new = cand["t_drop"]
    for c in chosen:
        if c["uav"]==u and abs(td_new - c["t_drop"]) < min_drop_gap:
            return False
    # 航向/速度锁
    thv = locks.get(u)
    if thv is not None:
        th_ref, v_ref = thv
        if _angdiff(cand["theta"], th_ref) > heading_tol_rad: return False
        if abs(cand["v"] - v_ref) > speed_tol: return False
    return True

def _marginal_gain_cover(cand, union_masks, tgrids, dt):
    gain = 0.0
    for name in tgrids:
        old = union_masks[name]
        new = np.logical_or(old, cand["mask_by_missile"][name])
        gain += float((new.sum() - old.sum()) * dt)
    return gain


# =========================
# 播种：保证 M1/M2/M3 均被覆盖
# =========================
def seed_cover_all_missiles(candidates: List[Dict[str,Any]],
                            tgrids: Dict[str,np.ndarray],
                            dt: float,
                            budgets_per_uav=BUDGETS,
                            min_drop_gap: float = MIN_DROP_GAP,
                            heading_tol_deg: float = HEADING_TOL_DEG,
                            speed_tol: float = SPEED_TOL):
    budgets = {i: budgets_per_uav[i] for i in range(len(UAVS))}
    locks: Dict[int, Optional[Tuple[float,float]]] = {i: None for i in range(len(UAVS))}
    union_masks = {name: np.zeros_like(tgrids[name], dtype=bool) for name in tgrids}
    chosen: List[Dict[str,Any]] = []
    tol_rad = math.radians(heading_tol_deg)

    for m in MISSILES:
        name = m["name"]
        best=None; best_gain=-1e18
        for c in candidates:
            if c["mask_by_missile"][name].sum() == 0:
                continue
            if not _feasible_with(c, chosen, budgets, min_drop_gap, locks, tol_rad, speed_tol):
                continue
            old = union_masks[name]
            new = np.logical_or(old, c["mask_by_missile"][name])
            gain = float((new.sum() - old.sum()) * dt)
            if gain > best_gain:
                best, best_gain = c, gain
        if best is not None and best_gain > 0.0:
            chosen.append(best)
            u = best["uav"]
            budgets[u] -= 1
            if locks[u] is None:
                locks[u] = (best["theta"], best["v"])
            for nm in tgrids:
                union_masks[nm] = np.logical_or(union_masks[nm], best["mask_by_missile"][nm])

    return chosen, union_masks, budgets, locks


# =========================
# 同航向同速度增产（固定 th/v 后，仅扫 t_burst 与 tau）
# =========================
def augment_locked_same_course(uav_idx: int, theta_ref: float, v_ref: float,
                               tgrids: Dict[str,np.ndarray], dt: float,
                               t_step: float = 1.0,
                               tau_list = (0.3,0.5,0.8,1.2,2.0,3.0,6.0,9.0,12.0),
                               dedup_eps: float = 0.10):
    u0 = UAVS[uav_idx]["U0"]
    T_hit_min = min(missile_hit_time(m["M0"]) for m in MISSILES)
    t_end = max(3.0, min(59.0, T_hit_min-0.5))
    t_bursts = np.arange(3.0, t_end+1e-9, t_step)

    cands=[]
    for t_b in t_bursts:
        for tau in tau_list:
            t_drop = t_b - tau
            if t_drop < 0.0 or t_drop > 60.0:
                continue
            cand = {"uav": uav_idx, "theta": theta_ref, "v": v_ref, "t_drop": t_drop, "tau": tau}
            sb = burst_point(u0, theta_ref, v_ref, t_drop, tau)
            if sb[2] <= 0.0:
                continue
            masks = _mask_for_candidate_L0(uav_idx, cand, tgrids)
            score = _cover_sum(masks, dt)
            cand.update({"mask_by_missile": masks, "score_cover": score, "t_burst": t_b})
            cands.append(cand)

    cands.sort(key=lambda c:c["t_burst"])
    filtered=[]
    for c in cands:
        if not filtered or abs(c["t_burst"]-filtered[-1]["t_burst"])>dedup_eps:
            filtered.append(c)
        else:
            if c["score_cover"] > filtered[-1]["score_cover"]:
                filtered[-1] = c
    filtered.sort(key=lambda c:c["score_cover"], reverse=True)
    return filtered


# =========================
# 贪心：在 seed 基础上继续加弹（优先边际覆盖），直到预算用完
# =========================
def greedy_fill_after_seed(candidates, tgrids, dt,
                           seed_chosen, seed_union, seed_budgets, seed_locks,
                           budgets_per_uav=BUDGETS,
                           min_drop_gap: float = MIN_DROP_GAP,
                           heading_tol_deg: float = HEADING_TOL_DEG,
                           speed_tol: float = SPEED_TOL):
    chosen = seed_chosen[:]
    union_masks = {nm: seed_union[nm].copy() for nm in tgrids}
    budgets = seed_budgets.copy()
    locks = seed_locks.copy()
    tol_rad = math.radians(heading_tol_deg)

    pool = candidates[:]
    for u in range(len(UAVS)):
        if locks[u] is not None:
            th_ref, v_ref = locks[u]
            extra = augment_locked_same_course(u, th_ref, v_ref, tgrids, dt, t_step=1.0)
            pool.extend(extra)

    while sum(max(0,b) for b in budgets.values()) > 0:
        best=None; best_gain=-1e18
        for c in pool:
            if not _feasible_with(c, chosen, budgets, min_drop_gap, locks, tol_rad, speed_tol):
                continue
            gain = _marginal_gain_cover(c, union_masks, tgrids, dt)
            if gain > best_gain:
                best, best_gain = c, gain
        if best is None:
            break
        chosen.append(best)
        u = best["uav"]
        budgets[u] -= 1
        if locks[u] is None:
            locks[u] = (best["theta"], best["v"])
            pool.extend(augment_locked_same_course(u, locks[u][0], locks[u][1], tgrids, dt, t_step=1.0))
        for nm in tgrids:
            union_masks[nm] = np.logical_or(union_masks[nm], best["mask_by_missile"][nm])
        pool = [x for x in pool if x is not best]
        if sum(max(0,b) for b in budgets.values()) == 0:
            break

    return chosen, union_masks, budgets, locks


# =========================
# 补齐 15 枚（允许遮蔽=0）
# =========================
def pad_zero_coverage_if_needed(chosen, union_masks, budgets, locks, tgrids, dt):
    """
    对于仍有剩余额度的 UAV，在其锁定的同航向/同速度下，以不同 t_drop/tau 生成“填充弹”，
    即使遮蔽=0 也添加，直到凑满 15 枚。间隔检查基于 t_drop。
    """
    for u in range(len(UAVS)):
        if locks.get(u) is None:
            u0 = UAVS[u]["U0"]
            theta = math.atan2(P_TARGET[1]-u0[1], P_TARGET[0]-u0[0])
            v = 100.0
            locks[u] = (theta, v)

    pools = {u: augment_locked_same_course(u, locks[u][0], locks[u][1], tgrids, dt, t_step=1.0) for u in range(len(UAVS))}

    def ok_interval(u, t_drop):
        for c in chosen:
            if c["uav"]==u and abs(c["t_drop"] - t_drop) < MIN_DROP_GAP:
                return False
        return True

    while sum(max(0,b) for b in budgets.values()) > 0:
        progressed=False
        for u in range(len(UAVS)):
            if budgets[u] <= 0:
                continue
            pick=None
            for c in pools[u]:
                if ok_interval(u, c["t_drop"]):
                    pick = c; break
            if pick is None:
                th, v = locks[u]
                t_try=0.0
                while t_try <= 60.0 and (not ok_interval(u, t_try)):
                    t_try += 0.6
                tau = 0.3
                cand = {"uav": u, "theta": th, "v": v, "t_drop": t_try, "tau": tau}
                cand["mask_by_missile"] = _mask_for_candidate_L0(u, cand, tgrids)
                cand["score_cover"] = _cover_sum(cand["mask_by_missile"], dt)
                cand["t_burst"] = t_try + tau
                pick = cand
            chosen.append(pick)
            budgets[u] -= 1
            for nm in tgrids:
                union_masks[nm] = np.logical_or(union_masks[nm], pick["mask_by_missile"][nm])
            progressed=True
            if sum(max(0,b) for b in budgets.values()) == 0:
                break
        if not progressed:
            break
    return chosen, union_masks


# =========================
# L1 小抛光（仅调整 t_drop/tau；保持同航向同速度；抛光也检测 t_drop 间隔）
# =========================
def polish_L1_keep_course(selected: List[Dict[str,Any]],
                          tgrids: Dict[str,np.ndarray],
                          dt_mask: float = DT_STEP,
                          rounds: int = POLISH_ROUNDS,
                          N_ang:int=48, N_Z:int=9, INCLUDE_SIDE:bool=True,
                          min_drop_gap: float = MIN_DROP_GAP):
    PTS = build_cylinder_samples(N_ang, N_Z, INCLUDE_SIDE)

    def rebuild_masks_L1(c):
        c2 = dict(c)
        c2["mask_by_missile"] = _mask_for_candidate_L1(c["uav"], c, tgrids, PTS)
        c2["score_cover"] = _cover_sum(c2["mask_by_missile"], dt_mask)
        return c2

    # 初始化：全部换成 L1 掩码
    cur = [rebuild_masks_L1(c) for c in selected]

    def score_of(sol):
        union = {nm: np.zeros_like(tgrids[nm], dtype=bool) for nm in tgrids}
        for c in sol:
            for nm in tgrids:
                union[nm] = np.logical_or(union[nm], c["mask_by_missile"][nm])
        sc = sum(float(union[nm].sum() * dt_mask) for nm in tgrids)
        return sc, union

    # ——【新增】抛光阶段的 t_drop 间隔约束检查——
    def ok_for_uav(u, td, others):
        for x in others:
            if x["uav"] == u and abs(x["t_drop"] - td) < min_drop_gap:
                return False
        return True

    best = cur[:]
    best_score, best_union = score_of(best)
    for _ in range(rounds):
        changed=False
        for i in range(len(best)):
            base = best[:i] + best[i+1:]
            c = best[i]
            # 小邻域（只动 t_drop、tau）；**先用 t_drop 做间隔过滤**
            tds  = [clip(c["t_drop"] + d, 0.0, 60.0) for d in (-1.2,-0.8,-0.4,0.0,0.4,0.8,1.2)]
            taus = [clip(c["tau"] * r, 0.2, 12.0)     for r in (0.85,1.0,1.15)]
            best_local=c; best_local_score=best_score
            for td in tds:
                if not ok_for_uav(c["uav"], td, base):   # ——这里按 t_drop 间隔过滤——
                    continue
                for ta in taus:
                    cand = dict(c)
                    cand["t_drop"]=td; cand["tau"]=ta; cand["t_burst"]=td+ta
                    cand = rebuild_masks_L1(cand)
                    sc,_ = score_of(base + [cand])
                    if sc > best_local_score:
                        best_local, best_local_score = cand, sc
            if best_local_score > best_score:
                best[i]=best_local; best_score=best_local_score; changed=True
        if not changed: break
    final_union = {nm: np.zeros_like(tgrids[nm], dtype=bool) for nm in tgrids}
    for c in best:
        for nm in tgrids:
            final_union[nm] = np.logical_or(final_union[nm], c["mask_by_missile"][nm])
    return best, final_union


# =========================
# 报表（12列；零遮蔽也输出一行）
# =========================
def _drop_point(u0: np.ndarray, theta: float, v_u: float, t_drop: float) -> np.ndarray:
    return uav_pos(u0, theta, v_u, t_drop)

def build_report_rows(selected_raw: List[Dict[str,Any]],
                      tgrids: Dict[str,np.ndarray],
                      dt: float) -> List[Dict[str,Any]]:
    # 各 UAV 内部按 t_drop 编号 FYi-1/2/3
    by_uav={}
    for c in selected_raw:
        by_uav.setdefault(c["uav"], []).append(c)
    for u,lst in by_uav.items():
        lst.sort(key=lambda x:x["t_drop"])
        for idx,c in enumerate(lst, start=1):
            c["_seq_in_uav"] = idx

    rows=[]
    selected_sorted = sorted(selected_raw, key=lambda c:(c["uav"], c["t_drop"]))
    for c in selected_sorted:
        u_idx=c["uav"]; u_name=UAVS[u_idx]["name"]; u0=UAVS[u_idx]["U0"]
        theta_deg=(math.degrees(c["theta"])%360.0); v_u=float(c["v"])
        t_drop=float(c["t_drop"]); tau=float(c["tau"])
        r_drop = _drop_point(u0, c["theta"], v_u, t_drop)
        s_burst = burst_point(u0, c["theta"], v_u, t_drop, tau)
        smoke_id=f"{u_name}-{c.get('_seq_in_uav',1)}"
        # 逐导弹输出（有正遮蔽的逐条；若都为0，输出一条 0）
        had_positive=False
        for m in MISSILES:
            name=m["name"]; mask=c["mask_by_missile"][name]
            eff_s=float(mask.sum()*dt)
            if eff_s<=0.0:
                continue
            had_positive=True
            rows.append({
                "无人机编号": u_name,
                "无人机运动方向": round(theta_deg,3),
                "无人机运动速度 (m/s)": round(v_u,3),
                "烟幕干扰弹编号": smoke_id,
                "烟幕干扰弹投放点的x坐标 (m)": round(float(r_drop[0]),3),
                "烟幕干扰弹投放点的y坐标 (m)": round(float(r_drop[1]),3),
                "烟幕干扰弹投放点的z坐标 (m)": round(float(r_drop[2]),3),
                "烟幕干扰弹起爆点的x坐标 (m)": round(float(s_burst[0]),3),
                "烟幕干扰弹起爆点的y坐标 (m)": round(float(s_burst[1]),3),
                "烟幕干扰弹起爆点的z坐标 (m)": round(float(s_burst[2]),3),
                "有效干扰时长 (s)": round(eff_s,3),
                "干扰的导弹编号": name
            })
        if not had_positive:
            rows.append({
                "无人机编号": u_name,
                "无人机运动方向": round(theta_deg,3),
                "无人机运动速度 (m/s)": round(v_u,3),
                "烟幕干扰弹编号": smoke_id,
                "烟幕干扰弹投放点的x坐标 (m)": round(float(r_drop[0]),3),
                "烟幕干扰弹投放点的y坐标 (m)": round(float(r_drop[1]),3),
                "烟幕干扰弹投放点的z坐标 (m)": round(float(r_drop[2]),3),
                "烟幕干扰弹起爆点的x坐标 (m)": round(float(s_burst[0]),3),
                "烟幕干扰弹起爆点的y坐标 (m)": round(float(s_burst[1]),3),
                "烟幕干扰弹起爆点的z坐标 (m)": round(float(s_burst[2]),3),
                "有效干扰时长 (s)": 0.0,
                "干扰的导弹编号": "-"
            })
    return rows

def print_report_rows(rows: List[Dict[str,Any]]):
    headers = [
        "无人机编号","无人机运动方向","无人机运动速度 (m/s)","烟幕干扰弹编号",
        "烟幕干扰弹投放点的x坐标 (m)","烟幕干扰弹投放点的y坐标 (m)","烟幕干扰弹投放点的z坐标 (m)",
        "烟幕干扰弹起爆点的x坐标 (m)","烟幕干扰弹起爆点的y坐标 (m)","烟幕干扰弹起爆点的z坐标 (m)",
        "有效干扰时长 (s)","干扰的导弹编号"
    ]
    print("\t".join(headers))
    for r in rows:
        print("\t".join(str(r[h]) for h in headers))


# =========================
# 主入口：先 L0，再 L1，凑满 15 枚
# =========================
def solve_q5(
    dt_step: float = DT_STEP,
    do_polish_L1: bool = True,
    heading_tol_deg: float = HEADING_TOL_DEG,
    speed_tol: float = SPEED_TOL
) -> Dict[str,Any]:
    # 1) L0 候选
    candidates_L0, tgrids_L0 = build_candidates_L0(dt=dt_step)
    counts = {i: sum(1 for c in candidates_L0 if c["uav"]==i) for i in range(len(UAVS))}
    print("[debug] L0 per-UAV nonzero candidates:", counts)

    # 2) 播种：保证三枚导弹都被至少一枚覆盖
    seed_chosen, seed_union, budgets, locks = seed_cover_all_missiles(
        candidates_L0, tgrids_L0, dt_step,
        budgets_per_uav=BUDGETS,
        min_drop_gap=MIN_DROP_GAP,
        heading_tol_deg=heading_tol_deg, speed_tol=speed_tol
    )

    # 3) 贪心继续加弹（允许边际=0）
    chosen, union, budgets, locks = greedy_fill_after_seed(
        candidates_L0, tgrids_L0, dt_step,
        seed_chosen, seed_union, budgets, locks,
        budgets_per_uav=BUDGETS,
        min_drop_gap=MIN_DROP_GAP,
        heading_tol_deg=heading_tol_deg, speed_tol=speed_tol
    )

    # 4) 补齐 15 枚（零遮蔽也允许）
    chosen, union = pad_zero_coverage_if_needed(chosen, union, budgets, locks, tgrids_L0, dt_step)

    # 安全裁剪到各 UAV 预算
    used_per_uav = {i:0 for i in range(len(UAVS))}
    final=[]
    for c in chosen:
        u=c["uav"]
        if used_per_uav[u] < BUDGETS[u]:
            final.append(c); used_per_uav[u]+=1

    total_need = sum(BUDGETS)
    if len(final) < total_need:
        for u in range(len(UAVS)):
            while used_per_uav[u] < BUDGETS[u]:
                u0 = UAVS[u]["U0"]; th,v = locks[u]
                t_drop = 0.0 + (used_per_uav[u])*0.6
                tau = 0.3
                c = {"uav": u, "theta": th, "v": v, "t_drop": t_drop, "tau": tau, "t_burst": t_drop+tau}
                c["mask_by_missile"] = _mask_for_candidate_L0(u, c, tgrids_L0)
                c["score_cover"] = _cover_sum(c["mask_by_missile"], dt_step)
                final.append(c); used_per_uav[u]+=1
                if len(final) >= total_need: break
        union = {nm: np.zeros_like(tgrids_L0[nm], dtype=bool) for nm in tgrids_L0}
        for c in final:
            for nm in tgrids_L0:
                union[nm] = np.logical_or(union[nm], c["mask_by_missile"][nm])

    chosen_L0, union_L0 = final, union

    # 5) L1 抛光（仅调 t_drop/tau；抛光也用 t_drop 间隔约束）
    mode = "L0"
    if do_polish_L1:
        tgrids_L1 = {m["name"]: _time_grid(missile_hit_time(m["M0"]), dt_step) for m in MISSILES}
        chosen_L1, union_L1 = polish_L1_keep_course(
            chosen_L0, tgrids_L1, dt_mask=dt_step,
            rounds=POLISH_ROUNDS, N_ang=N_ANG, N_Z=N_Z, INCLUDE_SIDE=INCLUDE_SIDE,
            min_drop_gap=MIN_DROP_GAP
        )
        chosen, union, tgrids = chosen_L1, union_L1, tgrids_L1
        mode = "L0 → L1(polish)"
    else:
        chosen, union, tgrids = chosen_L0, union_L0, tgrids_L0

    # 6) 汇总
    per_missile=[]
    for m in MISSILES:
        name=m["name"]; tgrid=tgrids[name]; mask=union[name]
        cover=float(mask.sum()*dt_step)
        intervals=[]; in_seg=False; a=None
        for k in range(len(mask)):
            if mask[k] and not in_seg: in_seg=True; a=float(tgrid[k])
            if in_seg and (k==len(mask)-1 or (not mask[k+1])):
                b=float(tgrid[k]); intervals.append((round(a,3), round(b,3))); in_seg=False
        per_missile.append({
            "missile": name,
            "T_hit": round(missile_hit_time(m["M0"]),3),
            "cover_s": round(cover,3),
            "intervals": intervals
        })
    cover_sum = round(sum(x["cover_s"] for x in per_missile), 3)

    out_sel=[]
    for c in chosen:
        u0 = UAVS[c["uav"]]["U0"]
        sb = burst_point(u0, c["theta"], c["v"], c["t_drop"], c["tau"])
        out_sel.append({
            "uav": UAVS[c["uav"]]["name"],
            "theta_deg": round(math.degrees(c["theta"]),3),
            "v_u_mps": round(float(c["v"]),3),
            "t_drop": round(float(c["t_drop"]),3),
            "tau": round(float(c["tau"]),3),
            "t_burst": round(float(c["t_drop"]+c["tau"]),3),
            "s_burst": (round(float(sb[0]),3), round(float(sb[1]),3), round(float(sb[2]),3))
        })

    rows = build_report_rows(chosen, tgrids, dt_step)

    used_total = len(chosen)
    if used_total < sum(BUDGETS):
        print(f"[warn] 总可行解不足：选了 {used_total}/{sum(BUDGETS)}。")

    return {
        "method": "graph_Q5",
        "mode": mode,
        "selected": out_sel,
        "per_missile": per_missile,
        "cover_sum_s": cover_sum,
        "used_total": used_total,
        "config": {
            "dt_step": dt_step,
            "budgets": BUDGETS,
            "min_drop_gap": MIN_DROP_GAP,
            "heading_tol_deg": HEADING_TOL_DEG,
            "speed_tol": SPEED_TOL,
            "L1_polish": do_polish_L1,
            "N_ANG": N_ANG, "N_Z": N_Z, "INCLUDE_SIDE": INCLUDE_SIDE
        },
        "rows": rows
    }


# =========================
# 示例运行
# =========================
if __name__ == "__main__":
    ans = solve_q5(dt_step=DT_STEP, do_polish_L1=True,
                   heading_tol_deg=HEADING_TOL_DEG, speed_tol=SPEED_TOL)

    print("\n[Q5 | 结果概览]")
    for k_, v in ans.items():
        if k_ != "rows":
            print(" ", k_, ":", v)

    print("\n[Q5 | 报表] 无人机×烟幕投放×导弹干扰明细（12列，含零遮蔽行）：")
    print_report_rows(ans["rows"])
