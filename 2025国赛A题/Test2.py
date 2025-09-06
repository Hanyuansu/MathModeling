# -*- coding: utf-8 -*-
"""
Q5：五架 UAV、每架 ≤3 枚，对三枚导弹的联合遮蔽（L0→L1→SA→Zero→Gain），用满 15 枚

特性与约束（全流程生效）：
- 每架 UAV 在“第一枚投放”时锁定 (theta, v)，之后该机所有投放必须同航向同速度（容差可设小）。
- 同一架飞机的投放时刻 t_drop 必须两两相隔 ≥ 1.0 s（候选生成、贪心、抛光、SA、零弹增益器都检测）。
- 统一时间步长 DT_STEP（L0/L1/统计完全一致）。
- 先 L0 生成候选，做“播种”（三导弹至少各被覆盖一次），再贪心加弹→补齐到 15 枚（必要时允许 0 覆盖填充）。
- L1 抛光（只动 t_drop/tau；不改 theta/v；仍检查 1s 间隔）。
- L1 模拟退火 SA（只动 t_drop/tau；不改 theta/v；仍检查 1s 间隔）。
- Zero→Gain Booster：把“当前覆盖=0”的弹，定向挪到三导弹的未覆盖空档附近，尽量带来正边际覆盖（仍检查 1s 间隔）。

输出：
- 结果概览（每导弹覆盖与时间区间、覆盖总秒数、用弹数量等）
- 12 列报表（无人机编号/航向/速度/投放点xyz/起爆点xyz/有效干扰时长/干扰导弹编号）
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

# 统一时间步长（L0/L1/统计一致）
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
MIN_DROP_GAP = 1.0      # 同机相邻投放最小间隔（全流程强制）

# 航向/速度锁定容差
HEADING_TOL_DEG = 2.0
SPEED_TOL = 1.5

# 候选生成参数（以导弹锚点反推）
FRACS  = (0.10, 0.18, 0.25, 0.40, 0.55, 0.70, 0.85, 0.92, 0.96, 0.985)
ALPHAS = (0.60, 0.70, 0.80, 0.88, 0.92, 0.96, 0.985)
TAUS_MUL = (0.55, 0.70, 0.85, 1.00, 1.15, 1.30)
PER_UAV_KEEP = 36
DEDUP_EPS = 0.12

# L1 目标采样
N_ANG, N_Z, INCLUDE_SIDE = 48, 9, True
POLISH_ROUNDS = 1        # L1 坐标下降轮次

# SA 参数（只动 t_drop/tau）
ENABLE_SA = True
SA_ITERS = 2000
SA_T0 = 1.0
SA_TEND = 1e-3
SA_PROB_SCALE = 1.0

# =========================
# 基本几何/运动学
# =========================
def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v));  return v/n if n>0 else v

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
    v = Q - P;  vv = float(np.dot(v, v))
    if vv==0.0: return float(np.linalg.norm(X-P))
    a = float(np.dot(X-P, v)/vv);  a = 0.0 if a<0 else (1.0 if a>1.0 else a)
    Y = P + a*v
    return float(np.linalg.norm(X - Y))

def clip(x, lo, hi): return lo if x<lo else (hi if x>hi else x)

# L0 判定：目标中心点代表视轴
def covered_L0_at_time(m0, p_target, s_burst, t_burst, t) -> bool:
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    return (point_to_segment_dist(p_target, m_t, s_t) <= R_SMOKE)

# L1 采样与判定
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
# 掩码/时间网格与打分
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
                        if score <= 0.0:
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

def _ok_min_gap_for_uav(uav_idx: int, t_drop: float, selected: List[Dict[str,Any]], min_gap: float) -> bool:
    for c in selected:
        if c["uav"]==uav_idx and abs(float(c["t_drop"])-t_drop) < min_gap-1e-9:
            return False
    return True

def _feasible_with(cand, chosen, budgets, min_drop_gap,
                   locks: Dict[int, Optional[Tuple[float,float]]],
                   heading_tol_rad: float, speed_tol: float) -> bool:
    u = cand["uav"]
    if budgets[u] <= 0: return False
    # 间隔
    if not _ok_min_gap_for_uav(u, cand["t_drop"], chosen, min_drop_gap):
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
# 首先：按导弹“播种”保证 M1/M2/M3 均被覆盖
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
# 贪心：在 seed 基础上继续加弹
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

    # 已锁 UAV 增产
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
    # 若某 UAV 未锁，给它一个“指向目标”的锁
    for u in range(len(UAVS)):
        if locks.get(u) is None:
            u0 = UAVS[u]["U0"]
            theta = math.atan2(P_TARGET[1]-u0[1], P_TARGET[0]-u0[0])
            v = 100.0
            locks[u] = (theta, v)
    # 为每架 UAV 生成同航向/同速度的“池”
    pools = {u: augment_locked_same_course(u, locks[u][0], locks[u][1], tgrids, dt, t_step=1.0) for u in range(len(UAVS))}

    def ok_interval(u, t_drop): return _ok_min_gap_for_uav(u, t_drop, chosen, MIN_DROP_GAP)

    while sum(max(0,b) for b in budgets.values()) > 0:
        progressed=False
        for u in range(len(UAVS)):
            if budgets[u] <= 0: continue
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
                cand = {"uav": u, "theta": th, "v": v, "t_drop": t_try, "tau": tau, "t_burst": t_try+tau}
                cand["mask_by_missile"] = _mask_for_candidate_L0(u, cand, tgrids)
                cand["score_cover"] = _cover_sum(cand["mask_by_missile"], dt)
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
# L1 抛光（仅调整 t_drop/tau；保持同航向同速度；检查1s间隔）
# =========================
def polish_L1_keep_course(selected: List[Dict[str,Any]],
                          tgrids: Dict[str,np.ndarray],
                          dt_mask: float = DT_STEP,
                          rounds: int = POLISH_ROUNDS,
                          N_ang:int=48, N_Z:int=9, INCLUDE_SIDE:bool=True):
    PTS = build_cylinder_samples(N_ang, N_Z, INCLUDE_SIDE)

    def rebuild_masks_L1(c):
        c2 = dict(c)
        c2["mask_by_missile"] = _mask_for_candidate_L1(c["uav"], c, tgrids, PTS)
        c2["score_cover"] = _cover_sum(c2["mask_by_missile"], dt_mask)
        return c2

    cur = [rebuild_masks_L1(c) for c in selected]

    def score_of(sol):
        union = {nm: np.zeros_like(tgrids[nm], dtype=bool) for nm in tgrids}
        for c in sol:
            for nm in tgrids:
                union[nm] = np.logical_or(union[nm], c["mask_by_missile"][nm])
        sc = sum(float(union[nm].sum() * dt_mask) for nm in tgrids)
        return sc, union

    best = cur[:]
    best_score, best_union = score_of(best)
    for _ in range(rounds):
        changed=False
        for i in range(len(best)):
            base = best[:i] + best[i+1:]
            c = best[i]
            tds  = [clip(c["t_drop"] + d, 0.0, 60.0) for d in (-1.2,-0.8,-0.4,0.0,0.4,0.8,1.2)]
            taus = [clip(c["tau"] * r, 0.2, 12.0)     for r in (0.85,1.0,1.15)]
            best_local=c; best_local_score=best_score
            for td in tds:
                # 1s 间隔检测
                if not _ok_min_gap_for_uav(c["uav"], td, base, MIN_DROP_GAP):
                    continue
                for ta in taus:
                    cand = dict(c)
                    cand["t_drop"]=td; cand["tau"]=ta; cand["t_burst"]=td+ta
                    cand = rebuild_masks_L1(cand)
                    sc,_ = score_of(base + [cand])
                    if sc > best_local_score + 1e-12:
                        best_local, best_local_score = cand, sc
            if best_local_score > best_score + 1e-12:
                best[i]=best_local; best_score=best_local_score; changed=True
        if not changed: break
    final_union = {nm: np.zeros_like(tgrids[nm], dtype=bool) for nm in tgrids}
    for c in best:
        for nm in tgrids:
            final_union[nm] = np.logical_or(final_union[nm], c["mask_by_missile"][nm])
    return best, final_union, best_score

# =========================
# L1 模拟退火（只动 t_drop/tau；保持航向/速度；检查1s间隔）
# =========================
def anneal_L1_keep_course(selected: List[Dict[str,Any]],
                          tgrids: Dict[str,np.ndarray],
                          dt: float = DT_STEP,
                          iters: int = SA_ITERS,
                          T0: float = SA_T0,
                          Tend: float = SA_TEND,
                          prob_scale: float = SA_PROB_SCALE,
                          N_ang:int=48, N_Z:int=9, INCLUDE_SIDE:bool=True,
                          seed: int = 2025):
    rng = np.random.default_rng(seed)
    PTS = build_cylinder_samples(N_ang, N_Z, INCLUDE_SIDE)

    def rebuild_masks(c):
        c2 = dict(c)
        c2["mask_by_missile"] = _mask_for_candidate_L1(c["uav"], c2, tgrids, PTS)
        c2["score_cover"] = _cover_sum(c2["mask_by_missile"], dt)
        return c2

    cur = [rebuild_masks(c) for c in selected]

    def score(sol):
        union = {nm: np.zeros_like(tgrids[nm], dtype=bool) for nm in tgrids}
        for c in sol:
            for nm in union:
                union[nm] = np.logical_or(union[nm], c["mask_by_missile"][nm])
        return float(sum(union[nm].sum() for nm in union) * dt), union

    best_score, best_union = score(cur)
    best_sol = [dict(c) for c in cur]
    cur_score, cur_union = best_score, best_union

    for k in range(iters):
        T = T0 * (Tend/T0)**(k/max(1, iters-1))
        i = rng.integers(0, len(cur))
        base = cur[:i] + cur[i+1:]
        c = cur[i]
        u = c["uav"]

        # 提案：小幅扰动 t_drop/tau
        td = float(c["t_drop"]) + rng.normal(0.0, 0.6)
        td = clip(td, 0.0, 60.0)
        if not _ok_min_gap_for_uav(u, td, base, MIN_DROP_GAP):
            continue
        ta = float(c["tau"]) * float(1.0 + rng.normal(0.0, 0.10))
        ta = clip(ta, 0.2, 12.0)
        # 高度与 T_HIT 检查
        t_burst = td + ta
        u0 = UAVS[u]["U0"]; sb = burst_point(u0, c["theta"], c["v"], td, ta)
        if sb[2] <= 0.0:
            continue
        ok_T=True
        for m in MISSILES:
            if t_burst >= missile_hit_time(m["M0"]):
                ok_T=False; break
        if not ok_T:
            continue

        cand = dict(c)
        cand["t_drop"]=td; cand["tau"]=ta; cand["t_burst"]=t_burst
        cand = rebuild_masks(cand)

        # 计算新的总覆盖（重用并集增量也可，这里简化求全）
        new_sol = base + [cand]
        new_score, _ = score(new_sol)
        delta = new_score - cur_score
        if delta >= 1e-12 or rng.random() < math.exp(delta/(max(1e-12,T)*prob_scale)):
            # 接受
            cur[i] = cand
            cur_score = new_score
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_sol = [dict(x) for x in cur]

    # 最终并集
    _, final_union = score(best_sol)
    return best_sol, final_union, best_score

# =========================
# Zero→Gain Booster：把零覆盖弹定向挪到空档
# =========================
def _build_union_from_selected_L1(selected: List[Dict[str,Any]],
                                  tgrids: Dict[str,np.ndarray],
                                  dt: float) -> Tuple[Dict[str,np.ndarray], float]:
    union = {nm: np.zeros_like(tgrids[nm], dtype=bool) for nm in tgrids}
    for c in selected:
        for nm in union:
            union[nm] = np.logical_or(union[nm], c["mask_by_missile"][nm])
    cover_sum = float(sum(union[nm].sum() for nm in union) * dt)
    return union, cover_sum

def _list_gaps(mask: np.ndarray, tgrid: np.ndarray, min_len: float) -> List[Tuple[float,float,Tuple[float,float]]]:
    gaps=[]; in_gap=False; a=None
    for k in range(len(mask)):
        if (not mask[k]) and (not in_gap):
            in_gap=True; a=float(tgrid[k])
        if in_gap and (k==len(mask)-1 or mask[k+1]):
            b=float(tgrid[k])
            if b-a >= min_len:
                gaps.append((0.5*(a+b), b-a, (a,b)))
            in_gap=False
    gaps.sort(key=lambda x:x[1], reverse=True)
    return gaps

def zero_salvage_gap_targeting_L1(
    selected_L1: List[Dict[str,Any]],
    tgrids: Dict[str,np.ndarray],
    dt: float,
    min_drop_gap: float = MIN_DROP_GAP,
    N_ang: int = 48, N_Z: int = 9, INCLUDE_SIDE: bool = True,
    max_outer_iter: int = 2,
    gap_scan_halfwin: float = 3.0,      # ±3s 搜索
    tburst_step: float = 0.30,          # t_burst 步长
    tau_list = (0.25,0.35,0.5,0.8,1.2,2.0,3.0,6.0,9.0),
) -> Tuple[List[Dict[str,Any]], Dict[str,np.ndarray], float]:
    PTS = build_cylinder_samples(N_ang, N_Z, INCLUDE_SIDE)

    def ensure_L1_masks(c):
        if "mask_by_missile" in c and isinstance(list(c["mask_by_missile"].values())[0], np.ndarray):
            return c
        c2 = dict(c)
        c2["mask_by_missile"] = _mask_for_candidate_L1(c["uav"], c2, tgrids, PTS)
        return c2

    selected = [ensure_L1_masks(c) for c in selected_L1]
    union, base_cover = _build_union_from_selected_L1(selected, tgrids, dt)

    def eff_s(c): return float(sum(c["mask_by_missile"][nm].sum() for nm in tgrids) * dt)
    def is_zero(c): return eff_s(c) <= 0.0 + 1e-12

    for _ in range(max_outer_iter):
        improved=False
        gaps_by_missile = {m["name"]: _list_gaps(union[m["name"]], tgrids[m["name"]], min_len=dt*2.0) for m in MISSILES}
        merged_gaps=[]
        for nm, gaps in gaps_by_missile.items():
            for center, length, ab in gaps:
                merged_gaps.append((nm, center, length, ab[0], ab[1]))
        merged_gaps.sort(key=lambda x:x[2], reverse=True)
        if not merged_gaps:
            break

        zero_ids = [i for i,c in enumerate(selected) if is_zero(c)]
        if not zero_ids:
            break

        for miss_name, center, glen, a, b in merged_gaps:
            tgrid = tgrids[miss_name]
            T_HIT = tgrid[-1]
            t_bursts = np.arange(max(0.2, center-gap_scan_halfwin),
                                 min(T_HIT-0.2, center+gap_scan_halfwin)+1e-9,
                                 tburst_step)
            for idx in list(zero_ids):
                c0 = selected[idx]
                u = c0["uav"]; th, v = float(c0["theta"]), float(c0["v"])
                best=None; best_gain=0.0

                base_without = selected[:idx]+selected[idx+1:]

                for t_b in t_bursts:
                    for tau in tau_list:
                        t_drop = t_b - tau
                        if t_drop < 0.0 or t_drop > 60.0:
                            continue
                        if not _ok_min_gap_for_uav(u, t_drop, base_without, min_drop_gap):
                            continue
                        sb = burst_point(UAVS[u]["U0"], th, v, t_drop, tau)
                        if sb[2] <= 0.0:
                            continue
                        cand = {"uav": u, "theta": th, "v": v, "t_drop": t_drop, "tau": tau, "t_burst": t_b}
                        cand["mask_by_missile"] = _mask_for_candidate_L1(u, cand, tgrids, PTS)
                        gain = 0.0
                        for nm in tgrids:
                            old = union[nm]
                            new = np.logical_or(old, cand["mask_by_missile"][nm])
                            gain += float((new.sum()-old.sum()) * dt)
                        if gain > best_gain + 1e-12:
                            best, best_gain = cand, gain

                if best is not None and best_gain > 0.0:
                    selected[idx] = best
                    for nm in tgrids:
                        union[nm] = np.logical_or(union[nm], best["mask_by_missile"][nm])
                    zero_ids.remove(idx)
                    improved=True

            if not zero_ids:
                break

        if not improved:
            break

    final_union, new_cover = _build_union_from_selected_L1(selected, tgrids, dt)
    return selected, final_union, new_cover

# =========================
# 报表（12列；零遮蔽也输出一行）
# =========================
def _drop_point(u0: np.ndarray, theta: float, v_u: float, t_drop: float) -> np.ndarray:
    return uav_pos(u0, theta, v_u, t_drop)

def build_report_rows(selected_raw: List[Dict[str,Any]],
                      tgrids: Dict[str,np.ndarray],
                      dt: float) -> List[Dict[str,Any]]:

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
# 主入口：L0 → L1 → SA → Zero→Gain
# =========================
def solve_q5(
    dt_step: float = DT_STEP,
    do_polish_L1: bool = True,
    do_SA: bool = ENABLE_SA,
    do_zero_gain: bool = True,
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

    # 4) 补齐 15 枚（必要时 0 覆盖填充；仍保持 1s 间隔）
    chosen, union = pad_zero_coverage_if_needed(chosen, union, budgets, locks, tgrids_L0, dt_step)

    # 5) L1 抛光（只动 t_drop/tau；保持航向/速度；检查 1s 间隔）
    mode = "L0"
    if do_polish_L1:
        tgrids_L1 = {m["name"]: _time_grid(missile_hit_time(m["M0"]), dt_step) for m in MISSILES}
        selected, union, score_L1 = polish_L1_keep_course(chosen, tgrids_L1, dt_mask=dt_step,
                                                          rounds=POLISH_ROUNDS, N_ang=N_ANG, N_Z=N_Z, INCLUDE_SIDE=INCLUDE_SIDE)
        mode = "L0 → L1(polish)"
    else:
        tgrids_L1 = tgrids_L0
        selected, score_L1 = chosen, float(sum(union[nm].sum() for nm in union) * dt_step)

    # 6) L1 模拟退火（可选）
    if do_SA:
        base_cover = float(sum(union[nm].sum() for nm in union) * dt_step)
        selected2, union2, score_SA = anneal_L1_keep_course(selected, tgrids_L1, dt=dt_step,
                                                             iters=SA_ITERS, T0=SA_T0, Tend=SA_TEND, prob_scale=SA_PROB_SCALE,
                                                             N_ang=N_ANG, N_Z=N_Z, INCLUDE_SIDE=INCLUDE_SIDE, seed=2025)
        if score_SA > base_cover + 1e-9:
            print(f"[info] SA 提升覆盖: {base_cover:.3f} → {score_SA:.3f} 秒")
        selected, union = selected2, union2
        mode += " → L1(SA)"

    # 7) Zero→Gain：把零覆盖弹挪到空档（可选）
    if do_zero_gain:
        base_cover = float(sum(union[nm].sum() for nm in union) * dt_step)
        selected3, union3, score_ZG = zero_salvage_gap_targeting_L1(
            selected, tgrids_L1, dt=dt_step,
            min_drop_gap=MIN_DROP_GAP,
            N_ang=N_ANG, N_Z=N_Z, INCLUDE_SIDE=INCLUDE_SIDE,
            max_outer_iter=2, gap_scan_halfwin=3.0, tburst_step=0.30,
            tau_list=(0.25,0.35,0.5,0.8,1.2,2.0,3.0,6.0,9.0)
        )
        if score_ZG > base_cover + 1e-9:
            print(f"[info] Zero→Gain 提升覆盖: {base_cover:.3f} → {score_ZG:.3f} 秒")
        selected, union = selected3, union3
        mode += " → L1(ZG)"

    # 8) 汇总
    per_missile=[]
    for m in MISSILES:
        name=m["name"]; tgrid=tgrids_L1[name]; mask=union[name]
        cover=float(mask.sum()*dt_step)
        intervals=[]; in_seg=False; a=None
        for k in range(len(mask)):
            if mask[k] and not in_seg: in_seg=True; a=float(tgrid[k])
            if in_seg and (k==len(mask)-1 or (not mask[k+1])):
                b=float(tgrid[k]); intervals.append((round(a,3), round(b,3))); in_seg=False
        per_missile.append({
            "missile": name,
            "T_hit": round(missile_hit_time(dict(MISSILES_map:= {m['name']:m for m in MISSILES})[name]["M0"]),3),
            "cover_s": round(cover,3),
            "intervals": intervals
        })
    cover_sum = round(sum(x["cover_s"] for x in per_missile), 3)

    # 结果条目
    out_sel=[]
    for c in selected:
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

    rows = build_report_rows(selected, tgrids_L1, dt_step)

    used_cnt = len(selected)
    if used_cnt != sum(BUDGETS):
        print(f"[warn] 已选择 {used_cnt}/{sum(BUDGETS)}，将尝试补齐/裁剪。")
    # 理论上已补齐 15 枚；若有差异可在此做守护（略）

    return {
        "method": "graph_Q5",
        "mode": mode,
        "selected": out_sel,
        "per_missile": per_missile,
        "cover_sum_s": cover_sum,
        "used_total": used_cnt,
        "config": {
            "dt_step": dt_step,
            "budgets": BUDGETS,
            "min_drop_gap": MIN_DROP_GAP,
            "heading_tol_deg": HEADING_TOL_DEG,
            "speed_tol": SPEED_TOL,
            "L1_polish": do_polish_L1,
            "L1_SA": do_SA, "SA_iters": SA_ITERS,
            "L1_ZG": do_zero_gain,
            "N_ANG": N_ANG, "N_Z": N_Z, "INCLUDE_SIDE": INCLUDE_SIDE
        },
        "rows": rows
    }

# =========================
# 示例运行
# =========================
if __name__ == "__main__":
    ans = solve_q5(dt_step=DT_STEP,
                   do_polish_L1=True,
                   do_SA=True,
                   do_zero_gain=True,
                   heading_tol_deg=HEADING_TOL_DEG,
                   speed_tol=SPEED_TOL)

    print("\n[Q5 | 结果概览]")
    for k_, v in ans.items():
        if k_ != "rows":
            print(" ", k_, ":", v)

    print("\n[Q5 | 报表] 无人机×烟幕投放×导弹干扰明细（12列，含零遮蔽行）：")
    print_report_rows(ans["rows"])
