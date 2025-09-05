# -*- coding: utf-8 -*-
"""
Q5（图论法，默认 L0；可选 L1 精修）—— 5 架 UAV、每架 ≤3 枚，联合干扰 M1/M2/M3

新增：将“间隔时间”纳入目标。
- 对每枚导弹 m，从并集掩码 mask_m 计算：
    覆盖时长 C_m（在首次覆盖至末次覆盖窗口内的 True * dt）
    间隔时长 G_m（同一窗口内的 False * dt）
- 目标：最大化 sum_m [ k*C_m - (1-k)*G_m ]，k∈[0,1] 由用户设定
  k=1 退化为只最大化覆盖；k 越小越倾向减少间隔、提升连续性。

新增（关键）：固定航向 + 同航向增产
- fixed_heading_per_uav=True：某 UAV 一旦确定首条投放航向，后续只能在该航向（容差内）投放
- augment_heading_locked_candidates：在首条锁航向后，沿此航向网格化扫描 t_burst/tau/v 生成更多“同航向候选”，
  保证每 UAV 能凑满 3 枚；配合 force_use_all=True 可以稳定用满 15 枚。

流程：候选生成 -> L0 掩码(三导弹) -> 分区+冲突约束子模贪心(固定航向+同航向增产+加权增益)
     -> gap定制兜底(同样用加权增益) -> （可选）L1 精修 -> 12列报表
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# =========================
# 常量与场景
# =========================
g = 9.81
VM = 300.0
V_SINK = 3.0
R_SMOKE = 10.0
T_EFFECT = 20.0

R_TAR, H_TAR = 7.0, 10.0
CYL_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)
P_TARGET   = np.array([0.0, 200.0, 5.0], dtype=float)

# 3 枚导弹初值
MISSILES = [
    {"name": "M1", "M0": np.array([20000.0,    0.0, 2000.0], dtype=float)},
    {"name": "M2", "M0": np.array([19000.0,  600.0, 2100.0], dtype=float)},
    {"name": "M3", "M0": np.array([18000.0, -600.0, 1900.0], dtype=float)},
]

# 5 架 UAV 初值
UAVS = [
    {"name": "FY1", "U0": np.array([17800.0,     0.0, 1800.0], dtype=float)},
    {"name": "FY2", "U0": np.array([12000.0,  1400.0, 1400.0], dtype=float)},
    {"name": "FY3", "U0": np.array([ 6000.0, -3000.0,  700.0], dtype=float)},
    {"name": "FY4", "U0": np.array([11000.0,  2000.0, 1800.0], dtype=float)},
    {"name": "FY5", "U0": np.array([13000.0, -2000.0, 1300.0], dtype=float)},
]

# =========================
# 几何与判定
# =========================
def missile_hit_time(m0: np.ndarray) -> float:
    return float(np.linalg.norm(m0) / VM)

def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v));  return v/n if n>0 else v

def missile_pos(m0: np.ndarray, t: float) -> np.ndarray:
    d = unit(-m0);  return m0 + VM * d * t

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
    a = float(np.dot(X-P, v)/vv);  a = 0.0 if a<0 else (1.0 if a>1 else a)
    Y = P + a*v
    return float(np.linalg.norm(X - Y))

def covered_L0_at_time(m0, p_target, s_burst, t_burst, t) -> bool:
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    return (point_to_segment_dist(p_target, m_t, s_t) <= R_SMOKE)

def clip(x, lo, hi): return lo if x<lo else (hi if x>hi else x)

# ---------- 可选 L1 判定 ----------
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
# 掩码统计：覆盖与间隔
# =========================
def _cover_and_gap(mask: np.ndarray, dt: float) -> Tuple[float, float]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0.0, 0.0
    a, b = int(idx[0]), int(idx[-1])
    win_true = int(mask[a:b+1].sum())
    win_len  = (b - a + 1)
    gap_cnt  = win_len - win_true
    C = win_true * dt
    G = gap_cnt  * dt
    return C, G

# =========================
# 候选生成（按导弹锚点）
# =========================
def _time_grid(T_hit: float, dt: float) -> np.ndarray:
    return np.arange(0.0, T_hit + 1e-12, dt)

def _candidate_from_anchor(u0: np.ndarray, m0: np.ndarray,
                           frac: float, alpha: float, tau_mult: float,
                           clamp_eps: float = 0.10):
    T_HIT = missile_hit_time(m0)
    t_b = clip(frac * min(60.0, T_HIT-2.0), 0.5, 59.5)
    m_tb = missile_pos(m0, t_b)
    Y = P_TARGET + alpha*(m_tb - P_TARGET)

    dx, dy = Y[0]-u0[0], Y[1]-u0[1]
    theta = math.atan2(dy, dx)
    D_xy  = math.hypot(dx, dy)
    v = clip(D_xy / t_b, 70.0, 140.0)

    u0z, Yz = float(u0[2]), float(Y[2])
    if u0z > Yz: tau_base = math.sqrt(max(0.0, 2.0*(u0z - Yz))/g)
    else:        tau_base = 0.2
    tau_max_by_t = max(0.2, t_b - clamp_eps)
    tau = clip(tau_base * tau_mult, 0.2, min(12.0, tau_max_by_t))
    t_drop = t_b - tau
    return theta%(2.0*math.pi), v, t_drop, tau

def _mask_for_candidate_L0(uav_idx: int, cand, tgrids: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    masks = {}
    u0 = UAVS[uav_idx]["U0"]
    theta, v, t_drop, tau = cand["theta"], cand["v"], cand["t_drop"], cand["tau"]
    t_burst = t_drop + tau
    for m in MISSILES:
        name, m0 = m["name"], m["M0"]
        T_HIT = missile_hit_time(m0)
        tgrid = tgrids[name]
        mask = np.zeros_like(tgrid, dtype=bool)
        if t_burst < T_HIT:
            s_burst = burst_point(u0, theta, v, t_drop, tau)
            if s_burst[2] > 0.0:
                t_start, t_end = t_burst, min(t_burst + T_EFFECT, T_HIT)
                idx = np.where((tgrid >= t_start) & (tgrid <= t_end))[0]
                for k in idx:
                    t = float(tgrid[k])
                    if covered_L0_at_time(m0, P_TARGET, s_burst, t_burst, t):
                        mask[k] = True
        masks[name] = mask
    return masks

def build_candidates_Q5(
    dt_mask: float = 0.015,
    fracs = (0.10,0.18,0.25,0.40,0.55,0.70,0.85,0.92,0.96,0.985),
    alphas= (0.60,0.70,0.80,0.88,0.92,0.96,0.985),
    taus  = (0.55,0.70,0.85,1.00,1.15,1.30),
    per_uav_keep: int = 36,
    dedup_eps: float = 0.12,
    k: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    tgrids = {m["name"]: _time_grid(missile_hit_time(m["M0"]), dt_mask) for m in MISSILES}
    all_cands=[]
    for uav_idx in range(len(UAVS)):
        local=[]
        for m in MISSILES:
            m0 = m["M0"]
            for f in fracs:
                for a in alphas:
                    for tau_mult in taus:
                        th, v, td, ta = _candidate_from_anchor(UAVS[uav_idx]["U0"], m0, f, a, tau_mult)
                        cand = {"uav": uav_idx, "theta": th, "v": v, "t_drop": td, "tau": ta}
                        masks = _mask_for_candidate_L0(uav_idx, cand, tgrids)
                        cover_sum=0.0; gap_sum=0.0
                        for name in tgrids:
                            C,G = _cover_and_gap(masks[name], dt_mask)
                            cover_sum += C; gap_sum += G
                        if cover_sum <= 0.0:
                            continue
                        score_k = k*cover_sum - (1.0-k)*gap_sum
                        cand.update({
                            "mask_by_missile": masks,
                            "score_sum_k": score_k,
                            "cover_sum": cover_sum,
                            "gap_sum": gap_sum,
                            "t_burst": td+ta
                        })
                        local.append(cand)
        # 轻度去重（按 t_burst 聚类）
        local.sort(key=lambda c:c["t_burst"])
        filtered=[]
        for c in local:
            if not filtered or abs(c["t_burst"] - filtered[-1]["t_burst"]) > dedup_eps:
                filtered.append(c)
            else:
                if c["score_sum_k"] > filtered[-1]["score_sum_k"]:
                    filtered[-1] = c
        filtered.sort(key=lambda c:c["score_sum_k"], reverse=True)
        all_cands += filtered[:per_uav_keep]
    return all_cands, tgrids

# =========================
# 固定航向 + 可行性
# =========================
def _ang_diff(a: float, b: float) -> float:
    d = abs((a - b + math.pi) % (2*math.pi) - math.pi)
    return d

def _feasible_with(
    cand: Dict[str,Any],
    chosen: List[Dict[str,Any]],
    budgets: Dict[int,int],
    min_drop_gap: float,
    fixed_heading_per_uav: bool,
    heading_tol_rad: float
) -> bool:
    u = cand["uav"]
    if budgets[u] <= 0: return False
    for c in chosen:
        if c["uav"] == u and abs(c["t_drop"] - cand["t_drop"]) < min_drop_gap:
            return False
    if fixed_heading_per_uav:
        theta_refs = [c["theta"] for c in chosen if c["uav"]==u]
        if theta_refs:
            if _ang_diff(cand["theta"], theta_refs[0]) > heading_tol_rad:
                return False
    return True

def _marginal_gain(cand, union_masks, tgrids, dt, k: float):
    gain = 0.0
    for name in tgrids:
        old_mask = union_masks[name]
        new_mask = np.logical_or(old_mask, cand["mask_by_missile"][name])
        C_old, G_old = _cover_and_gap(old_mask, dt)
        C_new, G_new = _cover_and_gap(new_mask, dt)
        gain += (k*(C_new - C_old) - (1.0 - k)*(G_new - G_old))
    return gain

# =========================
# ★ 同航向增产（关键）
# =========================
def augment_heading_locked_candidates(
    uav_idx: int,
    theta_ref: float,
    tgrids: Dict[str,np.ndarray],
    dt_mask: float,
    k: float,
    # 网格规模可适当调整；步长越密越容易“用满”
    t_burst_list = None,
    tau_list = (0.4, 0.8, 1.2, 2.0, 3.0, 6.0, 9.0, 12.0),
    v_list = (80.0, 95.0, 110.0, 125.0, 140.0),
    dedup_eps: float = 0.10
) -> List[Dict[str,Any]]:
    """
    在给定 UAV 的固定航向 theta_ref 上，扫描 t_burst/tau/v 生成更多候选。
    仅保留覆盖>0 的条目，并做简单去重（按 t_burst 聚类）。
    """
    u0 = UAVS[uav_idx]["U0"]
    # 允许的 t_burst 范围：从 3s 到所有导弹 T_HIT 的最小值-0.5（并限制在[3,59]）
    T_hit_min = min(missile_hit_time(m["M0"]) for m in MISSILES)
    if t_burst_list is None:
        # 步长 1.8s，覆盖 3~min(59, T_hit_min-0.5)
        t_end = max(3.0, min(59.0, T_hit_min-0.5))
        t_burst_list = np.arange(3.0, t_end+1e-9, 1.8)

    cands=[]
    for t_b in t_burst_list:
        for tau in tau_list:
            t_drop = t_b - tau
            if t_drop < 0.0 or t_drop > 60.0:  # 投放时刻约束
                continue
            for v in v_list:
                cand = {"uav": uav_idx, "theta": theta_ref, "v": v, "t_drop": t_drop, "tau": tau}
                # 起爆点 z>0 的快速判定（提升效率）
                sb = burst_point(u0, theta_ref, v, t_drop, tau)
                if sb[2] <= 0.0:
                    continue
                masks = _mask_for_candidate_L0(uav_idx, cand, tgrids)
                cover_sum=0.0; gap_sum=0.0
                for name in tgrids:
                    C,G = _cover_and_gap(masks[name], dt_mask)
                    cover_sum += C; gap_sum += G
                if cover_sum <= 0.0:
                    continue
                score_k = k*cover_sum - (1.0-k)*gap_sum
                cand.update({
                    "mask_by_missile": masks,
                    "score_sum_k": score_k,
                    "cover_sum": cover_sum,
                    "gap_sum": gap_sum,
                    "t_burst": t_b
                })
                cands.append(cand)

    # 按 t_burst 轻度去重
    cands.sort(key=lambda c:c["t_burst"])
    filtered=[]
    for c in cands:
        if not filtered or abs(c["t_burst"] - filtered[-1]["t_burst"]) > dedup_eps:
            filtered.append(c)
        else:
            if c["score_sum_k"] > filtered[-1]["score_sum_k"]:
                filtered[-1] = c
    # 适度按分数降序（利于后续贪心）
    filtered.sort(key=lambda c:c["score_sum_k"], reverse=True)
    return filtered

# =========================
# 贪心选择（带同航向增产）
# =========================
def greedy_select_Q5(
    candidates: List[Dict[str, Any]],
    tgrids: Dict[str, np.ndarray],
    budgets_per_uav = (3,3,3,3,3),
    min_drop_gap: float = 1.0,
    dt_mask: float = 0.015,
    k: float = 1.0,
    fixed_heading_per_uav: bool = True,
    heading_tol_deg: float = 2.0,
    force_use_all: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    budgets = {i: budgets_per_uav[i] for i in range(len(UAVS))}
    union_masks = {name: np.zeros_like(tgrids[name], dtype=bool) for name in tgrids}
    chosen: List[Dict[str, Any]] = []
    remain = candidates[:]
    total_budget = sum(budgets.values())
    tol_rad = math.radians(heading_tol_deg)

    # 统计：用于检测“首条触发增产”
    picked_count_per_uav = {i:0 for i in range(len(UAVS))}

    while len(chosen) < total_budget:
        best, best_gain = None, -1e18
        for c in remain:
            if not _feasible_with(c, chosen, budgets, min_drop_gap, fixed_heading_per_uav, tol_rad):
                continue
            gain = _marginal_gain(c, union_masks, tgrids, dt_mask, k)
            if gain > best_gain:
                best, best_gain = c, gain

        if best is None:
            break
        if (not force_use_all) and best_gain <= 0.0:
            break

        # 接受
        u = best["uav"]
        before = picked_count_per_uav[u]
        chosen.append(best)
        picked_count_per_uav[u] += 1
        budgets[u] -= 1
        for name in tgrids:
            union_masks[name] = np.logical_or(union_masks[name], best["mask_by_missile"][name])
        remain = [x for x in remain if x is not best]

        # ★ 若该 UAV 首次被选中，立刻在该航向上“增产”同航向候选并加入 remain
        if fixed_heading_per_uav and before == 0:
            theta_ref = best["theta"]
            extra = augment_heading_locked_candidates(
                uav_idx=u,
                theta_ref=theta_ref,
                tgrids=tgrids,
                dt_mask=dt_mask,
                k=k
            )
            # 避免把“已不可能可行”的候选全塞进来：先做一个粗过滤（与已选的该 UAV 投放间隔 >= min_drop_gap）
            # 这样可以减少无谓评估的数量
            def ok_interval(cnew):
                for csel in chosen:
                    if csel["uav"]==u and abs(csel["t_drop"] - cnew["t_drop"]) < min_drop_gap:
                        return False
                return True
            extra = [e for e in extra if ok_interval(e)]
            remain.extend(extra)

    return chosen, union_masks

# ---------- 最大空档（每导弹） ----------
def _largest_gap(mask: np.ndarray, tgrid: np.ndarray) -> Tuple[float, float]:
    if len(tgrid) < 2: return 0.0, 0.0
    gaps=[]; in_gap=False; a=None
    for k in range(len(mask)):
        if (not mask[k]) and (not in_gap):
            in_gap=True; a=float(tgrid[k])
        if in_gap and (k==len(mask)-1 or mask[k+1]):
            b=float(tgrid[k]); gaps.append((a,b)); in_gap=False
    if not gaps: return float(tgrid[len(tgrid)//2]), 0.0
    a,b = max(gaps, key=lambda it: it[1]-it[0])
    return 0.5*(a+b), (b-a)

def _synthesize_for_gap_Q5(uav_idx: int, missile_idx: int, center_t: float,
                           dt_mask: float, tgrids: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
    m0 = MISSILES[missile_idx]["M0"]
    u0 = UAVS[uav_idx]["U0"]
    T_HIT = missile_hit_time(m0)
    t_b = clip(center_t, 0.6, min(59.5, T_HIT-0.5))
    frac = t_b / min(60.0, T_HIT-2.0)

    alphas_try = (0.92, 0.96, 0.985)
    tau_mult_try = (0.70, 0.85, 1.00, 1.15)

    def mk(theta, v, td, ta):
        cand = {"uav": uav_idx, "theta":theta, "v":v, "t_drop":td, "tau":ta}
        masks = _mask_for_candidate_L0(uav_idx, cand, tgrids)
        cover_sum=0.0; gap_sum=0.0
        for nm in tgrids:
            C,G = _cover_and_gap(masks[nm], dt_mask)
            cover_sum += C; gap_sum += G
        cand.update({"mask_by_missile": masks, "score_sum_k": cover_sum - gap_sum, "cover_sum": cover_sum, "gap_sum": gap_sum, "t_burst": td+ta})
        return cand

    best=None
    th0,v0,td0,ta0 = _candidate_from_anchor(u0, m0, frac, 0.96, 1.0)
    yawN = [th0 + math.radians(d) for d in (-10,-6,-3,0,3,6,10)]
    vN   = [clip(v0 + dv, 70.0, 140.0) for dv in (-14,-8,0,8,14)]
    tdN  = [clip(td0 + d, 0.0, 60.0)   for d in (-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5)]
    for a in alphas_try:
        th,v,td,ta = _candidate_from_anchor(u0, m0, frac, a, 1.0)
        for th1 in yawN:
            for v1 in vN:
                for td1 in tdN:
                    for mul in (0.70, 0.85, 1.00, 1.15):
                        ta1 = clip(ta*mul, 0.2, min(12.0, t_b-0.1))
                        c1 = mk(th1, v1, td1, ta1)
                        if (best is None) or (c1["score_sum_k"] > best["score_sum_k"]):
                            best = c1
    return best

def gap_fill_phase(
    chosen: List[Dict[str,Any]],
    union_masks: Dict[str,np.ndarray],
    candidates: List[Dict[str,Any]],
    tgrids: Dict[str,np.ndarray],
    dt_mask: float,
    budgets_per_uav,
    min_drop_gap: float,
    k: float,
    fixed_heading_per_uav: bool = True,
    heading_tol_deg: float = 2.0,
    force_use_all: bool = True
):
    budgets = {i: budgets_per_uav[i] - sum(1 for c in chosen if c["uav"]==i) for i in range(len(UAVS))}
    tol_rad = math.radians(heading_tol_deg)
    for m_idx, _ in enumerate(MISSILES):
        center_t, gap = _largest_gap(union_masks[MISSILES[m_idx]["name"]], tgrids[MISSILES[m_idx]["name"]])
        if gap <= 0.0:
            continue
        improved=True; trial_times=0
        while improved and trial_times < 3:
            improved=False; trial_times += 1
            best=None; best_gain = -1e18
            for u in range(len(UAVS)):
                if budgets[u] <= 0: continue
                synth = _synthesize_for_gap_Q5(u, m_idx, center_t, dt_mask, tgrids)
                if synth is None: continue
                if not _feasible_with(synth, chosen, budgets, min_drop_gap, fixed_heading_per_uav, tol_rad):
                    # 若固定航向且已锁航向，可考虑把 synth 的航向替换为锁定航向再重算掩码（此处省略，通常前面的增产已足够）
                    continue
                gain = _marginal_gain(synth, union_masks, tgrids, dt_mask, k)
                if gain > best_gain:
                    best, best_gain = synth, gain
            if best is not None and (force_use_all or best_gain > 0.0):
                chosen.append(best)
                budgets[best["uav"]] -= 1
                for nm in tgrids:
                    union_masks[nm] = np.logical_or(union_masks[nm], best["mask_by_missile"][nm])
                improved=True
    return chosen, union_masks

# =========================
# 可选 L1 精修（小邻域坐标下降）
# =========================
def polish_L1(selected: List[Dict[str, Any]], tgrids: Dict[str, np.ndarray],
              dt_mask=0.02, N_ANG=48, N_Z=9, INCLUDE_SIDE=True, rounds=2):
    PTS = build_cylinder_samples(N_ang=N_ANG, N_Z=N_Z, include_side=INCLUDE_SIDE)
    def mask_L1_for(c):
        u0 = UAVS[c["uav"]]["U0"]
        theta, v, td, ta = c["theta"], c["v"], c["t_drop"], c["tau"]
        t_burst = td + ta
        s_burst = burst_point(u0, theta, v, td, ta)
        masks={}
        for m in MISSILES:
            name, m0 = m["name"], m["M0"]
            T_HIT = missile_hit_time(m0)
            mask = np.zeros_like(tgrids[name], dtype=bool)
            if (s_burst[2] > 0.0) and (t_burst < T_HIT):
                t_start, t_end = t_burst, min(t_burst+T_EFFECT, T_HIT)
                idx = np.where((tgrids[name] >= t_start) & (tgrids[name] <= t_end))[0]
                for k in idx:
                    t = float(tgrids[name][k])
                    if covered_L1_at_time_vectorized(m0, s_burst, t_burst, t, PTS):
                        mask[k]=True
            masks[name]=mask
        return masks
    def score_of(sol):
        union = {nm: np.zeros_like(tgrids[nm], dtype=bool) for nm in tgrids}
        for c in sol:
            for nm in tgrids:
                union[nm] = np.logical_or(union[nm], c["mask_by_missile"][nm])
        sc = sum(float(union[nm].sum()*dt_mask) for nm in tgrids)
        return sc, union
    for c in selected:
        c["mask_by_missile"] = mask_L1_for(c)
    best = selected[:]
    best_score, _ = score_of(best)
    for _ in range(rounds):
        changed=False
        for i in range(len(best)):
            base = best[:i] + best[i+1:]
            c = best[i]
            thetas = [c["theta"] + math.radians(d) for d in (-8,-4,0,4,8)]
            vs     = [clip(c["v"] + dv, 70.0, 140.0) for dv in (-12,-6,0,6,12)]
            tds    = [clip(c["t_drop"] + dt, 0.0, 60.0) for dt in (-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5)]
            taus   = [clip(c["tau"] * r, 0.2, 12.0) for r in (0.85,1.0,1.15)]
            best_local=c; best_local_score=best_score
            for th in thetas:
                for v in vs:
                    for td in tds:
                        for ta in taus:
                            cand = {"uav": c["uav"], "theta": th, "v": v, "t_drop": td, "tau": ta}
                            cand["mask_by_missile"] = mask_L1_for(cand)
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
# 报表（12列）
# =========================
def _drop_point(u0: np.ndarray, theta: float, v_u: float, t_drop: float) -> np.ndarray:
    return uav_pos(u0, theta, v_u, t_drop)

def build_report_rows(selected_raw: List[Dict[str, Any]],
                      tgrids: Dict[str, np.ndarray],
                      dt_mask: float) -> List[Dict[str, Any]]:
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
        theta_deg = (math.degrees(c["theta"])%360.0); v_u=c["v"]; t_drop=c["t_drop"]; tau=c["tau"]
        r_drop = _drop_point(u0, c["theta"], v_u, t_drop)
        s_burst = burst_point(u0, c["theta"], v_u, t_drop, tau)
        smoke_id = f"{u_name}-{c.get('_seq_in_uav',1)}"
        for m in MISSILES:
            name = m["name"]; mask = c["mask_by_missile"][name]
            eff_s = float(mask.sum())*dt_mask
            if eff_s <= 0.0: continue
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
    return rows

def print_report_rows(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "无人机编号","无人机运动方向","无人机运动速度 (m/s)",
        "烟幕干扰弹编号",
        "烟幕干扰弹投放点的x坐标 (m)","烟幕干扰弹投放点的y坐标 (m)","烟幕干扰弹投放点的z坐标 (m)",
        "烟幕干扰弹起爆点的x坐标 (m)","烟幕干扰弹起爆点的y坐标 (m)","烟幕干扰弹起爆点的z坐标 (m)",
        "有效干扰时长 (s)","干扰的导弹编号"
    ]
    print("\t".join(headers))
    for r in rows:
        print("\t".join(str(r[h]) for h in headers))

# =========================
# 求解入口
# =========================
def solve_q5_graph(
    dt_mask: float = 0.015,
    budgets_per_uav = (3,3,3,3,3),
    min_drop_gap: float = 1.0,
    fracs=(0.10,0.18,0.25,0.40,0.55,0.70,0.85,0.92,0.96,0.985),
    alphas=(0.60,0.70,0.80,0.88,0.92,0.96,0.985),
    taus=(0.55,0.70,0.85,1.00,1.15,1.30),
    per_uav_keep=36,
    do_gap_fill: bool = True,
    do_polish_L1: bool = False,
    N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
    k: float = 1.0,                       # 覆盖-间隔权重
    fixed_heading_per_uav: bool = True,
    heading_tol_deg: float = 2.0,
    force_use_all: bool = True
) -> Dict[str, Any]:
    # 1) 候选（按 k 排序）
    candidates, tgrids = build_candidates_Q5(dt_mask, fracs, alphas, taus, per_uav_keep, k=k)
    counts = {i: sum(1 for c in candidates if c["uav"]==i) for i in range(len(UAVS))}
    print("[debug] per-UAV nonzero candidates:", counts)

    # 2) 贪心选择（加权边际增益 + 固定航向 + 同航向增产 + 是否用满）
    chosen, union_masks = greedy_select_Q5(
        candidates, tgrids,
        budgets_per_uav=budgets_per_uav,
        min_drop_gap=min_drop_gap,
        dt_mask=dt_mask,
        k=k,
        fixed_heading_per_uav=fixed_heading_per_uav,
        heading_tol_deg=heading_tol_deg,
        force_use_all=force_use_all
    )

    # 3) gap 兜底
    if do_gap_fill:
        chosen, union_masks = gap_fill_phase(
            chosen, union_masks, candidates, tgrids, dt_mask,
            budgets_per_uav, min_drop_gap, k,
            fixed_heading_per_uav=fixed_heading_per_uav,
            heading_tol_deg=heading_tol_deg,
            force_use_all=force_use_all
        )

    # 4) L1 精修（可选）
    if do_polish_L1:
        chosen, union_masks = polish_L1(chosen, tgrids, dt_mask, N_ANG, N_Z, INCLUDE_SIDE)

    # 5) 统计
    per_missile=[]; score_k_total=0.0; gap_sum_total=0.0
    for m in MISSILES:
        name = m["name"]; tgrid = tgrids[name]; mask = union_masks[name]
        cover = float(mask.sum()*dt_mask)
        C_m, G_m = _cover_and_gap(mask, dt_mask)
        gap_sum_total += G_m
        score_k_total += (k*C_m - (1.0-k)*G_m)

        intervals=[]; in_seg=False; a=None
        for kk in range(len(mask)):
            if mask[kk] and not in_seg: in_seg=True; a=float(tgrid[kk])
            if in_seg and (kk==len(mask)-1 or (not mask[kk+1])):
                b=float(tgrid[kk]); intervals.append((round(a,3), round(b,3))); in_seg=False
        per_missile.append({
            "missile": name,
            "T_hit": round(missile_hit_time(m["M0"]), 3),
            "cover_s": round(cover, 3),
            "gap_s":   round(G_m, 3),
            "intervals": intervals
        })

    cover_sum = sum(x["cover_s"] for x in per_missile)
    worst_gap = 0.0
    for m in MISSILES:
        name = m["name"]; tgrid = tgrids[name]
        _, gap = _largest_gap(union_masks[name], tgrid)
        worst_gap = max(worst_gap, gap)

    out_sel=[]
    for c in chosen:
        u0 = UAVS[c["uav"]]["U0"]
        sb = burst_point(u0, c["theta"], c["v"], c["t_drop"], c["tau"])
        out_sel.append({
            "uav": UAVS[c["uav"]]["name"],
            "theta_deg": round(math.degrees(c["theta"]), 3),
            "v_u_mps": round(c["v"], 3),
            "t_drop": round(c["t_drop"], 3),
            "tau": round(c["tau"], 3),
            "t_burst": round(c["t_drop"]+c["tau"], 3),
            "s_burst": (round(float(sb[0]),3), round(float(sb[1]),3), round(float(sb[2]),3))
        })

    rows = build_report_rows(chosen, tgrids, dt_mask)

    return {
        "method": "graph_Q5",
        "mode": f"L0{' + L1_polish' if do_polish_L1 else ''}",
        "selected": out_sel,
        "rows": rows,
        "per_missile": per_missile,
        "cover_sum_s": round(cover_sum, 3),
        "gap_sum_total_s": round(gap_sum_total, 3),
        "score_k_total": round(score_k_total, 3),
        "worst_gap_s": round(worst_gap, 3),
        "config": {
            "dt_mask": dt_mask, "fracs": fracs, "alphas": alphas, "taus": taus,
            "per_uav_keep": per_uav_keep, "budgets": budgets_per_uav,
            "gap_fill": do_gap_fill, "L1_polish": do_polish_L1,
            "N_ANG": N_ANG, "N_Z": N_Z, "INCLUDE_SIDE": INCLUDE_SIDE,
            "k": k,
            "fixed_heading_per_uav": fixed_heading_per_uav,
            "heading_tol_deg": heading_tol_deg,
            "force_use_all": force_use_all
        }
    }

# =========================
# 主程序（示例）
# =========================
if __name__ == "__main__":
    ans = solve_q5_graph(
        dt_mask=0.015,
        budgets_per_uav=(3,3,3,3,3),
        min_drop_gap=1.0,
        do_gap_fill=True,
        do_polish_L1=False,
        k=1.0,                       # 纯覆盖 → 单调不减，更易用满
        fixed_heading_per_uav=True,  # ★ 固定航向
        heading_tol_deg=2.0,
        force_use_all=True           # ★ 用满 15 枚
    )

    print("\n[Q5 | Graph] 结果摘要：")
    for k_, v in ans.items():
        if k_ != "rows":
            print(" ", k_, ":", v)

    print("\n[Q5 | 报表] 无人机×烟幕投放×导弹干扰明细（12列）：")
    print_report_rows(ans["rows"])
