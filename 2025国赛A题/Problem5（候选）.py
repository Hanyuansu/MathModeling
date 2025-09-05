# -*- coding: utf-8 -*-
"""
Q5（图论法，支持 L0 / L1 / two_stage 三策略）—— 5 架 UAV、每架 ≤3 枚，联合干扰 M1/M2/M3
流程：候选生成 -> 掩码(三导弹) -> 分区+冲突约束子模贪心 -> gap定制兜底 -> (可选)小抛光
- strategy='L0'       ：全程用 L0 判定（快）
- strategy='L1'       ：全程用 L1 判定（准）
- strategy='two_stage'：L0 海选 + 精选后 L1 复评与选择（折中）
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from itertools import permutations

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

# 3 枚导弹初值（可按题面调整）
MISSILES = [
    {"name": "M1", "M0": np.array([20000.0,    0.0, 2000.0], dtype=float)},
    {"name": "M2", "M0": np.array([19000.0,  600.0, 2100.0], dtype=float)},
    {"name": "M3", "M0": np.array([18000.0, -600.0, 1900.0], dtype=float)},
]

# 5 架 UAV 初值（可按题面调整）
UAVS = [
    {"name": "FY1", "U0": np.array([17800.0,     0.0, 1800.0], dtype=float)},
    {"name": "FY2", "U0": np.array([12000.0,  1400.0, 1400.0], dtype=float)},
    {"name": "FY3", "U0": np.array([ 6000.0, -3000.0,  700.0], dtype=float)},
    {"name": "FY4", "U0": np.array([11000.0,  2000.0, 1800.0], dtype=float)},
    {"name": "FY5", "U0": np.array([13000.0, -2000.0, 1300.0], dtype=float)},
]

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

# ---------- L1 采样与判定 ----------
def cyl_points_top_bottom(N_ang: int = 48) -> np.ndarray:
    cx, cy, cz = CYL_CENTER; out=[]
    for z in (cz, cz + H_TAR):
        for k in range(N_ang):
            ang = 2.0*math.pi*k/N_ang
            out.append((cx+R_TAR*math.cos(ang), cy+R_TAR*math.sin(ang), z))
    return np.array(out, dtype=float)

def cyl_points_side(N_ang: int = 48, N_z: int = 9) -> np.ndarray:
    cx, cy, cz = CYL_CENTER; zs = np.linspace(cz, cz + H_TAR, N_z); out=[]
    for z in zs:
        for k in range(N_ang):
            ang = 2.0*math.pi*k/N_ang
            out.append((cx+R_TAR*math.cos(ang), cy+R_TAR*math.sin(ang), z))
    return np.array(out, dtype=float)

def build_cylinder_samples(N_ang=48, N_z=9, include_side=True) -> np.ndarray:
    pts=[cyl_points_top_bottom(N_ang)]
    if include_side: pts.append(cyl_points_side(N_ang, N_z))
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
# 候选生成（按导弹锚点；每候选对3枚导弹都计算掩码）
# =========================
def _time_grid(T_hit: float, dt: float) -> np.ndarray:
    return np.arange(0.0, T_hit + 1e-12, dt)

def _candidate_from_anchor(u0: np.ndarray, m0: np.ndarray,
                           frac: float, alpha: float, tau_mult: float,
                           clamp_eps: float = 0.10):
    """
    用“某枚导弹”的时刻锚点生成一枚候选（物理参数与导弹无关，但锚点靠这枚导弹引导）
    """
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

def _mask_for_candidate(uav_idx: int,
                        cand: Dict[str, Any],
                        tgrids: Dict[str, np.ndarray],
                        mask_mode: str = "L0",
                        PTS: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    统一的掩码计算器：mask_mode ∈ {'L0','L1'}
    """
    masks = {}
    u0 = UAVS[uav_idx]["U0"]
    theta, v, t_drop, tau = cand["theta"], cand["v"], cand["t_drop"], cand["tau"]
    t_burst = t_drop + tau
    for m in MISSILES:
        name, m0 = m["name"], m["M0"]
        T_HIT = missile_hit_time(m0)
        tgrid = tgrids[name]
        mask = np.zeros_like(tgrid, dtype=bool)

        if t_burst >= T_HIT:
            masks[name] = mask;  continue

        s_burst = burst_point(u0, theta, v, t_drop, tau)
        if s_burst[2] <= 0.0:
            masks[name] = mask;  continue

        t_start, t_end = t_burst, min(t_burst + T_EFFECT, T_HIT)
        idx = np.where((tgrid >= t_start) & (tgrid <= t_end))[0]

        if mask_mode == "L0":
            for k in idx:
                t = float(tgrid[k])
                if covered_L0_at_time(m0, P_TARGET, s_burst, t_burst, t):
                    mask[k] = True
        else:
            assert PTS is not None, "L1 模式需要提供 PTS 采样点"
            for k in idx:
                t = float(tgrid[k])
                if covered_L1_at_time_vectorized(m0, s_burst, t_burst, t, PTS):
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
    mask_mode: str = "L0",
    PTS: Optional[np.ndarray] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    为每个 UAV 生成候选；按 mask_mode 计算掩码与分数（对三枚导弹求和）
    """
    # 为每枚导弹构造时间网格
    tgrids = {m["name"]: _time_grid(missile_hit_time(m["M0"]), dt_mask) for m in MISSILES}

    all_cands = []
    for uav_idx in range(len(UAVS)):
        u0 = UAVS[uav_idx]["U0"]
        local = []
        for m in MISSILES:
            m0 = m["M0"]
            for f in fracs:
                for a in alphas:
                    for tau_mult in taus:
                        th, v, td, ta = _candidate_from_anchor(u0, m0, f, a, tau_mult)
                        cand = {"uav": uav_idx, "theta": th, "v": v, "t_drop": td, "tau": ta}
                        masks = _mask_for_candidate(uav_idx, cand, tgrids, mask_mode, PTS)
                        score = sum(float(masks[name].sum() * dt_mask) for name in tgrids)
                        if score <= 0.0:
                            continue
                        cand.update({
                            "mask_by_missile": masks,
                            "score_sum": score,
                            "t_burst": td + ta
                        })
                        local.append(cand)
        # 轻度去重（按 t_burst 聚类）
        local.sort(key=lambda c: c["t_burst"])
        filtered = []
        for c in local:
            if not filtered or abs(c["t_burst"] - filtered[-1]["t_burst"]) > dedup_eps:
                filtered.append(c)
            else:
                if c["score_sum"] > filtered[-1]["score_sum"]:
                    filtered[-1] = c
        # 只留前K
        filtered.sort(key=lambda c: c["score_sum"], reverse=True)
        all_cands += filtered[:per_uav_keep]
    return all_cands, tgrids

# =========================
# 选择（子模贪心 + 冲突/配额）
# =========================
def _feasible_with(cand, chosen, budgets, min_drop_gap=1.0) -> bool:
    u = cand["uav"]
    if budgets[u] <= 0: return False
    for c in chosen:
        if c["uav"] == u and abs(c["t_drop"] - cand["t_drop"]) < min_drop_gap:
            return False
    return True

def _marginal_gain(cand, union_masks, tgrids, dt):
    gain = 0.0
    for name in tgrids:
        new_union = np.logical_or(union_masks[name], cand["mask_by_missile"][name])
        gain += float((new_union.sum() - union_masks[name].sum()) * dt)
    return gain

def greedy_select_Q5(
    candidates: List[Dict[str, Any]],
    tgrids: Dict[str, np.ndarray],
    budgets_per_uav = (3,3,3,3,3),
    min_drop_gap: float = 1.0,
    dt_mask: float = 0.015,
    max_total: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    budgets = {i: budgets_per_uav[i] for i in range(len(UAVS))}
    union_masks = {name: np.zeros_like(tgrids[name], dtype=bool) for name in tgrids}
    chosen: List[Dict[str, Any]] = []

    remain = candidates[:]
    total_budget = sum(budgets.values()) if max_total is None else max_total

    while len(chosen) < total_budget:
        best, best_gain = None, 0.0
        for c in remain:
            if not _feasible_with(c, chosen, budgets, min_drop_gap):
                continue
            gain = _marginal_gain(c, union_masks, tgrids, dt_mask)
            if gain > best_gain:
                best, best_gain = c, gain
        if best is None or best_gain <= 0.0:
            break
        chosen.append(best)
        budgets[best["uav"]] -= 1
        for name in tgrids:
            union_masks[name] = np.logical_or(union_masks[name], best["mask_by_missile"][name])
        remain = [x for x in remain if x is not best]
    return chosen, union_masks

# ---------- 最大空档与 gap 填充 ----------
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
                           dt_mask: float, tgrids: Dict[str, np.ndarray],
                           mask_mode: str = "L0", PTS: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
    """
    针对某 UAV + 某导弹在 gap 中心附近“合成一枚”并做小邻域抛光（按策略 L0/L1）
    """
    m0 = MISSILES[missile_idx]["M0"]
    u0 = UAVS[uav_idx]["U0"]
    T_HIT = missile_hit_time(m0)
    t_b = clip(center_t, 0.6, min(59.5, T_HIT-0.5))
    frac = t_b / min(60.0, T_HIT-2.0)

    alphas_try = (0.92, 0.96, 0.985)
    tau_mult_try = (0.70, 0.85, 1.00, 1.15)

    def mk(theta, v, td, ta):
        cand = {"uav": uav_idx, "theta":theta, "v":v, "t_drop":td, "tau":ta}
        masks = _mask_for_candidate(uav_idx, cand, tgrids, mask_mode, PTS)
        score = sum(float(masks[nm].sum() * dt_mask) for nm in tgrids)
        cand.update({"mask_by_missile": masks, "score_sum": score, "t_burst": td+ta})
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
                    for mul in tau_mult_try:
                        ta1 = clip(ta*mul, 0.2, min(12.0, t_b-0.1))
                        c1 = mk(th1, v1, td1, ta1)
                        if (best is None) or (c1["score_sum"] > best["score_sum"]):
                            best = c1
    return best

def gap_fill_phase(chosen, union_masks, tgrids, dt_mask,
                   budgets_per_uav, min_drop_gap=1.0,
                   mask_mode: str = "L0", PTS: Optional[np.ndarray] = None):
    budgets = {i: budgets_per_uav[i] - sum(1 for c in chosen if c["uav"]==i) for i in range(len(UAVS))}
    # 依次处理每枚导弹最大的未覆盖空档
    for m_idx, m in enumerate(MISSILES):
        name = m["name"]
        center_t, gap = _largest_gap(union_masks[name], tgrids[name])
        if gap <= 0.0:  # 没空档
            continue
        improved = True; trial_times=0
        while improved and trial_times < 3:
            improved = False; trial_times += 1
            best=None; best_gain=0.0
            for u in range(len(UAVS)):
                if budgets[u] <= 0: continue
                synth = _synthesize_for_gap_Q5(u, m_idx, center_t, dt_mask, tgrids, mask_mode, PTS)
                if synth is None: continue
                if not _feasible_with(synth, chosen, budgets, min_drop_gap): continue
                gain = _marginal_gain(synth, union_masks, tgrids, dt_mask)
                if gain > best_gain:
                    best, best_gain = synth, gain
            if best is not None and best_gain > 0.0:
                chosen.append(best)
                budgets[best["uav"]] -= 1
                for nm in tgrids:
                    union_masks[nm] = np.logical_or(union_masks[nm], best["mask_by_missile"][nm])
                improved = True
    return chosen, union_masks

# =========================
# 可选 L1 抛光（坐标下降在小邻域）
# =========================
def polish_L1(selected: List[Dict[str, Any]], tgrids: Dict[str, np.ndarray],
              dt_mask=0.02, N_ANG=48, N_Z=9, INCLUDE_SIDE=True, rounds=2):
    PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)

    def mask_L1_for(c):
        return _mask_for_candidate(c["uav"], c, tgrids, "L1", PTS)

    def score_of(sol):
        union = {nm: np.zeros_like(tgrids[nm], dtype=bool) for nm in tgrids}
        for c in sol:
            for nm in tgrids:
                union[nm] = np.logical_or(union[nm], c["mask_by_missile"][nm])
        sc = sum(float(union[nm].sum()*dt_mask) for nm in tgrids)
        return sc, union

    # 初始化用 L1 掩码重算
    for c in selected:
        c["mask_by_missile"] = mask_L1_for(c)

    best = selected[:]
    best_score, best_union = score_of(best)

    for _ in range(rounds):
        changed=False
        for i in range(len(best)):
            base = best[:i] + best[i+1:]
            c = best[i]
            thetas = [c["theta"] + math.radians(d) for d in (-8,-4,0,4,8)]
            vs     = [clip(c["v"] + dv, 70.0, 140.0) for dv in (-12,-6,0,6,12)]
            tds    = [clip(c["t_drop"] + dt, 0.0, 60.0) for dt in (-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5)]
            taus   = [clip(c["tau"] * r, 0.2, 12.0) for r in (0.85,1.0,1.15)]
            best_local = c; best_local_score = best_score
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
                best[i] = best_local
                best_score = best_local_score
                changed=True
        if not changed: break
    final_union = {nm: np.zeros_like(tgrids[nm], dtype=bool) for nm in tgrids}
    for c in best:
        for nm in tgrids:
            final_union[nm] = np.logical_or(final_union[nm], c["mask_by_missile"][nm])
    return best, final_union

# =========================
# 主入口：一键切换策略
# =========================
def solve_q5_graph(
    strategy: str = "L0",        # 'L0' | 'L1' | 'two_stage'
    dt_mask: float = 0.015,
    budgets_per_uav = (3,3,3,3,3),
    min_drop_gap: float = 1.0,
    fracs=(0.10,0.18,0.25,0.40,0.55,0.70,0.85,0.92,0.96,0.985),
    alphas=(0.60,0.70,0.80,0.88,0.92,0.96,0.985),
    taus=(0.55,0.70,0.85,1.00,1.15,1.30),
    per_uav_keep=36,
    # two_stage 模式专用：L0 海选规模与 L1 精选规模
    dual_keep_L0_per_uav: int = 48,
    dual_refine_L1_per_uav: int = 18,
    # L1/two_stage 采样参数
    N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
    do_gap_fill: bool = True,
    do_polish_L1: bool = False   # L1 或 two_stage 下可开（会慢一些）
) -> Dict[str, Any]:
    # backward-compat: 接受历史参数 'dual'
    if strategy == "dual":
        strategy = "two_stage"
    assert strategy in ("L0","L1","two_stage")

    # 预构建 L1 采样（仅在需要时）
    PTS = None
    if strategy in ("L1","two_stage") or do_polish_L1:
        PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)

    # ---------- 1) 候选 & 掩码 ----------
    if strategy == "L0":
        candidates, tgrids = build_candidates_Q5(
            dt_mask, fracs, alphas, taus, per_uav_keep, 0.12, "L0", None
        )
    elif strategy == "L1":
        candidates, tgrids = build_candidates_Q5(
            dt_mask, fracs, alphas, taus, per_uav_keep, 0.12, "L1", PTS
        )
    elif strategy == "two_stage":  # L0 海选→取每 UAV 前 keep，再用 L1 复评这些候选
        cands_L0, tgrids = build_candidates_Q5(
            dt_mask, fracs, alphas, taus, dual_keep_L0_per_uav, 0.12, "L0", None
        )
        # 按 UAV 分组，取各自前 dual_refine_L1_per_uav 做 L1 复评
        grouped={i:[] for i in range(len(UAVS))}
        for c in cands_L0: grouped[c["uav"]].append(c)
        candidates=[]
        for u in range(len(UAVS)):
            top = sorted(grouped[u], key=lambda x:x["score_sum"], reverse=True)[:dual_refine_L1_per_uav]
            for c in top:
                masks = _mask_for_candidate(u, c, tgrids, "L1", PTS)
                score = sum(float(masks[name].sum()*dt_mask) for name in tgrids)
                c_new = dict(c)
                c_new["mask_by_missile"]=masks
                c_new["score_sum"]=score
                candidates.append(c_new)
        # 再按分数截到 per_uav_keep
        grouped={i:[] for i in range(len(UAVS))}
        for c in candidates: grouped[c["uav"]].append(c)
        candidates=[]
        for u in range(len(UAVS)):
            candidates += sorted(grouped[u], key=lambda x:x["score_sum"], reverse=True)[:per_uav_keep]

    counts = {i: sum(1 for c in candidates if c["uav"]==i) for i in range(len(UAVS))}
    print(f"[debug] strategy={strategy} per-UAV candidates:", counts)

    # ---------- 2) 贪心选择（子模 + 分区 + 冲突） ----------
    chosen, union_masks = greedy_select_Q5(
        candidates, tgrids, budgets_per_uav, min_drop_gap, dt_mask
    )

    # ---------- 3) gap 填充（按策略） ----------
    if do_gap_fill:
        chosen, union_masks = gap_fill_phase(
            chosen, union_masks, tgrids, dt_mask, budgets_per_uav, min_drop_gap,
            ("L1" if strategy in ("L1","two_stage") else "L0"), PTS
        )

    # ---------- 4) 可选 L1 抛光 ----------
    if do_polish_L1:
        chosen, union_masks = polish_L1(chosen, tgrids, dt_mask, N_ANG, N_Z, INCLUDE_SIDE)

    # ---------- 5) 统计与输出 ----------
    per_missile = []
    for m in MISSILES:
        name = m["name"]; tgrid = tgrids[name]
        mask = union_masks[name]
        cover = float(mask.sum() * dt_mask)
        intervals=[]; in_seg=False; a=None
        for k in range(len(mask)):
            if mask[k] and not in_seg: in_seg=True; a=float(tgrid[k])
            if in_seg and (k==len(mask)-1 or (not mask[k+1])):
                b=float(tgrid[k]); intervals.append((round(a,3), round(b,3))); in_seg=False
        per_missile.append({
            "missile": name,
            "T_hit": round(missile_hit_time(m["M0"]), 3),
            "cover_s": round(cover, 3),
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

    return {
        "method": "graph_Q5",
        "mode": strategy + (" + L1_polish" if do_polish_L1 else ""),
        "selected": out_sel,
        "per_missile": per_missile,
        "cover_sum_s": cover_sum,
        "worst_gap_s": round(worst_gap, 3),
        "config": {
            "strategy": strategy,
            "dt_mask": dt_mask, "fracs": fracs, "alphas": alphas, "taus": taus,
            "per_uav_keep": per_uav_keep, "budgets": budgets_per_uav,
            "gap_fill": do_gap_fill, "polish_L1": do_polish_L1,
            "N_ANG": N_ANG, "N_Z": N_Z, "INCLUDE_SIDE": INCLUDE_SIDE,
            "dual_keep_L0_per_uav": dual_keep_L0_per_uav,
            "dual_refine_L1_per_uav": dual_refine_L1_per_uav
        }
    }


# =========================
# 示例
# =========================
if __name__ == "__main__":
    # 任选其一：'L0' / 'L1' / 'two_stage'
    ans = solve_q5_graph(
        strategy="two_stage",   # <-- 现在改成 'two_stage'；仍兼容传入 'dual'
        dt_mask=0.015,
        budgets_per_uav=(3,3,3,3,3),
        min_drop_gap=1.0,
        do_gap_fill=True,
        do_polish_L1=True,     # L1/two_stage 下如需更稳可 True（会更慢）
        # two_stage 专用规模参数（名字沿用 dual_*，避免不必要的大改）
        dual_keep_L0_per_uav=48,
        dual_refine_L1_per_uav=18
    )
    print("\n[Q5 | Graph] 结果：")
    for k,v in ans.items():
        print(" ", k, ":", v)
