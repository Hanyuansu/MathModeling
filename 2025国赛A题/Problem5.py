# -*- coding: utf-8 -*-
"""
Q5（图论法，默认 L0；可选 L1 精修）—— 5 架 UAV、每架 ≤3 枚，联合干扰 M1/M2/M3

新增：将“间隔时间”纳入目标。
- 对每枚导弹 m，从并集掩码 mask_m 计算：
    覆盖时长 C_m（在首次覆盖至末次覆盖窗口内的 True * dt）
    间隔时长 G_m（同一窗口内的 False * dt）
- 目标：最大化 sum_m [ k*C_m - (1-k)*G_m ]，k∈[0,1] 由用户设定
  k=1 退化为只最大化覆盖；k 越小越倾向减少间隔、提升连续性。

流程：候选生成 -> L0 掩码(对三枚导弹) -> 分区+冲突约束下的子模贪心(用加权增益)
     -> gap定制兜底(同样用加权增益) -> （可选）L1 精修
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

def build_cylinder_samples(N_ang=48, N_z=9, include_side=True) -> np.ndarray:
    pts=[cyl_points_top_bottom(N_ang)];
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
# 掩码统计：覆盖与间隔
# =========================
def _cover_and_gap(mask: np.ndarray, dt: float) -> Tuple[float, float]:
    """
    从布尔掩码计算：
      C: 覆盖时长（在首次 True 到末次 True 的窗口内，True 计入）
      G: 间隔时长（同一窗口内，False 计入）
    若无覆盖则 (0.0, 0.0)
    """
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

def _mask_for_candidate_L0(uav_idx: int, cand, tgrids: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    对三枚导弹各自的时间网格计算 L0 掩码
    """
    masks = {}
    u0 = UAVS[uav_idx]["U0"]
    theta, v, t_drop, tau = cand["theta"], cand["v"], cand["t_drop"], cand["tau"]
    t_burst = t_drop + tau
    for m in MISSILES:
        name, m0 = m["name"], m["M0"]
        T_HIT = missile_hit_time(m0)
        if t_burst >= T_HIT:
            masks[name] = np.zeros_like(tgrids[name], dtype=bool);  continue
        s_burst = burst_point(u0, theta, v, t_drop, tau)
        if s_burst[2] <= 0.0:
            masks[name] = np.zeros_like(tgrids[name], dtype=bool);  continue
        t_start, t_end = t_burst, min(t_burst + T_EFFECT, T_HIT)
        tgrid = tgrids[name]
        mask = np.zeros_like(tgrid, dtype=bool)
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
    k: float = 1.0,                 # ← 覆盖-间隔权重
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    为每个 UAV 生成候选（用每枚导弹作锚点引导），并对三枚导弹都计算掩码
    去重：同一 UAV 内 t_burst 很接近（< eps）且参数近似的只留一个
    候选打分：score_sum_k = k*sum(C_m) - (1-k)*sum(G_m)
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
                        masks = _mask_for_candidate_L0(uav_idx, cand, tgrids)

                        cover_sum = 0.0
                        gap_sum   = 0.0
                        for name in tgrids:
                            C, G = _cover_and_gap(masks[name], dt_mask)
                            cover_sum += C
                            gap_sum   += G
                        score_k = k*cover_sum - (1.0 - k)*gap_sum

                        if cover_sum <= 0.0:
                            continue

                        cand.update({
                            "mask_by_missile": masks,
                            "score_sum_k": score_k,      # 用于排序
                            "cover_sum": cover_sum,      # 记录以作分析
                            "gap_sum": gap_sum,
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
                if c["score_sum_k"] > filtered[-1]["score_sum_k"]:
                    filtered[-1] = c

        # 只留前K（按新的加权目标排序）
        filtered.sort(key=lambda c: c["score_sum_k"], reverse=True)
        all_cands += filtered[:per_uav_keep]
    return all_cands, tgrids

# =========================
# 选择（子模贪心 + 冲突/配额）
# =========================
def _feasible_with(cand, chosen, budgets, min_drop_gap=1.0) -> bool:
    u = cand["uav"]
    if budgets[u] <= 0: return False
    # 投放间隔：同一 UAV 的 t_drop 至少间隔 1s
    for c in chosen:
        if c["uav"] == u and abs(c["t_drop"] - cand["t_drop"]) < min_drop_gap:
            return False
    return True

def _marginal_gain(cand, union_masks, tgrids, dt, k: float):
    """
    加权边际增益：Δ[k*C - (1-k)*G]，对所有导弹求和
    """
    gain = 0.0
    for name in tgrids:
        old_mask = union_masks[name]
        new_mask = np.logical_or(old_mask, cand["mask_by_missile"][name])

        C_old, G_old = _cover_and_gap(old_mask, dt)
        C_new, G_new = _cover_and_gap(new_mask, dt)
        gain += (k*(C_new - C_old) - (1.0 - k)*(G_new - G_old))
    return gain

def greedy_select_Q5(
    candidates: List[Dict[str, Any]],
    tgrids: Dict[str, np.ndarray],
    budgets_per_uav = (3,3,3,3,3),
    min_drop_gap: float = 1.0,
    dt_mask: float = 0.015,
    max_total: Optional[int] = None,
    k: float = 1.0,                        # ← 新增：权重
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
            gain = _marginal_gain(c, union_masks, tgrids, dt_mask, k)
            if gain > best_gain:
                best, best_gain = c, gain
        if best is None or best_gain <= 0.0:
            break
        # 接受
        chosen.append(best)
        budgets[best["uav"]] -= 1
        for name in tgrids:
            union_masks[name] = np.logical_or(union_masks[name], best["mask_by_missile"][name])
        # 移除这条候选，减少计算
        remain = [x for x in remain if x is not best]
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
    """
    针对某 UAV + 某导弹在 gap 中心附近“合成一枚”并做小邻域抛光（纯 L0）
    """
    m0 = MISSILES[missile_idx]["M0"]; name = MISSILES[missile_idx]["name"]
    u0 = UAVS[uav_idx]["U0"]
    T_HIT = missile_hit_time(m0)
    t_b = clip(center_t, 0.6, min(59.5, T_HIT-0.5))
    frac = t_b / min(60.0, T_HIT-2.0)

    alphas_try = (0.92, 0.96, 0.985)
    tau_mult_try = (0.70, 0.85, 1.00, 1.15)

    def mk(theta, v, td, ta):
        cand = {"uav": uav_idx, "theta":theta, "v":v, "t_drop":td, "tau":ta}
        masks = _mask_for_candidate_L0(uav_idx, cand, tgrids)
        # 记录覆盖/间隔（排序不使用，这里仅便于诊断）
        cover_sum = 0.0
        gap_sum   = 0.0
        for nm in tgrids:
            C, G = _cover_and_gap(masks[nm], dt_mask)
            cover_sum += C
            gap_sum   += G
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
                    for mul in tau_mult_try:
                        ta1 = clip(ta*mul, 0.2, min(12.0, t_b-0.1))
                        c1 = mk(th1, v1, td1, ta1)
                        if (best is None) or (c1["score_sum_k"] > best["score_sum_k"]):
                            best = c1
    return best

def gap_fill_phase(chosen, union_masks, candidates, tgrids, dt_mask, budgets_per_uav, min_drop_gap=1.0, k: float = 1.0):
    budgets = {i: budgets_per_uav[i] - sum(1 for c in chosen if c["uav"]==i) for i in range(len(UAVS))}
    # 依次处理每枚导弹最大的未覆盖空档
    for m_idx, m in enumerate(MISSILES):
        name = m["name"]
        center_t, gap = _largest_gap(union_masks[name], tgrids[name])
        if gap <= 0.0:  # 没空档
            continue
        # 尝试为“有剩余额度的 UAV”合成并加入
        improved = True; trial_times=0
        while improved and trial_times < 3:
            improved = False; trial_times += 1
            best=None; best_gain=0.0
            for u in range(len(UAVS)):
                if budgets[u] <= 0: continue
                synth = _synthesize_for_gap_Q5(u, m_idx, center_t, dt_mask, tgrids)
                if synth is None: continue
                if not _feasible_with(synth, chosen, budgets, min_drop_gap): continue
                gain = _marginal_gain(synth, union_masks, tgrids, dt_mask, k)
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
# 可选 L1 精修（坐标下降在小邻域）
# =========================
def polish_L1(selected: List[Dict[str, Any]], tgrids: Dict[str, np.ndarray],
              dt_mask=0.02, N_ANG=48, N_Z=9, INCLUDE_SIDE=True, rounds=2):
    PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)

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
            if (s_burst[2] <= 0.0) or (t_burst >= T_HIT):
                masks[name]=mask; continue
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

    # 初始化用 L1 掩码重算
    for c in selected:
        c["mask_by_missile"] = mask_L1_for(c)

    best = selected[:]
    best_score, best_union = score_of(best)

    for _ in range(rounds):
        changed=False
        for i in range(len(best)):
            base = best[:i] + best[i+1:]
            # 小邻域
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
                            trial = base + [cand]
                            sc,_ = score_of(trial)
                            if sc > best_local_score:
                                best_local, best_local_score = cand, sc
            if best_local_score > best_score:
                best[i] = best_local
                best_score = best_local_score
                changed=True
        if not changed: break
    # 最终并集
    final_union = {nm: np.zeros_like(tgrids[nm], dtype=bool) for nm in tgrids}
    for c in best:
        for nm in tgrids:
            final_union[nm] = np.logical_or(final_union[nm], c["mask_by_missile"][nm])
    return best, final_union

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
    k: float = 1.0,                       # ← 覆盖-间隔权重
) -> Dict[str, Any]:
    # 1) 候选（已按 k 排序）
    candidates, tgrids = build_candidates_Q5(dt_mask, fracs, alphas, taus, per_uav_keep, k=k)
    counts = {i: sum(1 for c in candidates if c["uav"]==i) for i in range(len(UAVS))}
    print("[debug] per-UAV nonzero candidates:", counts)

    # 2) 贪心选择（子模 + 分区 + 冲突）—— 使用加权边际增益
    chosen, union_masks = greedy_select_Q5(
        candidates, tgrids, budgets_per_uav, min_drop_gap, dt_mask, k=k
    )

    # 3) gap 填充兜底（可选）
    if do_gap_fill:
        chosen, union_masks = gap_fill_phase(
            chosen, union_masks, candidates, tgrids, dt_mask, budgets_per_uav, min_drop_gap, k=k
        )

    # 4) L1 精修（可选）
    if do_polish_L1:
        chosen, union_masks = polish_L1(chosen, tgrids, dt_mask, N_ANG, N_Z, INCLUDE_SIDE)

    # 统计
    per_missile = []
    score_k_total = 0.0
    gap_sum_total = 0.0
    for m in MISSILES:
        name = m["name"]; tgrid = tgrids[name]
        mask = union_masks[name]
        cover = float(mask.sum() * dt_mask)
        C_m, G_m = _cover_and_gap(mask, dt_mask)
        gap_sum_total += G_m
        score_k_total += (k*C_m - (1.0 - k)*G_m)

        # 由掩码恢复区间
        intervals=[]; in_seg=False; a=None
        for kk in range(len(mask)):
            if mask[kk] and not in_seg: in_seg=True; a=float(tgrid[kk])
            if in_seg and (kk==len(mask)-1 or (not mask[kk+1])):
                b=float(tgrid[kk]); intervals.append((round(a,3), round(b,3))); in_seg=False
        per_missile.append({
            "missile": name,
            "T_hit": round(missile_hit_time(m["M0"]), 3),
            "cover_s": round(cover, 3),
            "gap_s":   round(G_m, 3),          # 新增：窗口内间隔总时长
            "intervals": intervals
        })

    cover_sum = sum(x["cover_s"] for x in per_missile)
    # 最差空档（求最大未覆盖长度的最大值）
    worst_gap = 0.0
    for m in MISSILES:
        name = m["name"]; tgrid = tgrids[name]
        center, gap = _largest_gap(union_masks[name], tgrid)
        worst_gap = max(worst_gap, gap)

    # 整理输出（含起爆几何）
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
        "mode": f"L0{' + L1_polish' if do_polish_L1 else ''}",
        "selected": out_sel,
        "per_missile": per_missile,
        "cover_sum_s": round(cover_sum, 3),
        "gap_sum_total_s": round(gap_sum_total, 3),   # 新增：总间隔时长
        "score_k_total": round(score_k_total, 3),     # 新增：加权目标值
        "worst_gap_s": round(worst_gap, 3),
        "config": {
            "dt_mask": dt_mask, "fracs": fracs, "alphas": alphas, "taus": taus,
            "per_uav_keep": per_uav_keep, "budgets": budgets_per_uav,
            "gap_fill": do_gap_fill, "L1_polish": do_polish_L1,
            "N_ANG": N_ANG, "N_Z": N_Z, "INCLUDE_SIDE": INCLUDE_SIDE,
            "k": k
        }
    }

# =========================
# 主程序（示例）
# =========================
if __name__ == "__main__":
    # # 1) 纯覆盖优先（k=1.0，与原版一致）
    # ans = solve_q5_graph(
    #     dt_mask=0.015,
    #     budgets_per_uav=(3, 3, 3, 3, 3),
    #     min_drop_gap=1.0,
    #     do_gap_fill=True,
    #     do_polish_L1=False,
    #     k=1.0
    # )
    # print("\n[Q5 | Graph, k=1.0] 结果：")
    # for k_, v in ans.items():
    #     print(" ", k_, ":", v)
    #
    # # 2) 同时考虑“覆盖↑、间隔↓”：例如 k=0.8
    # ans_k = solve_q5_graph(
    #     dt_mask=0.015,
    #     budgets_per_uav=(3,3,3,3,3),
    #     min_drop_gap=1.0,
    #     do_gap_fill=True,
    #     do_polish_L1=False,
    #     k=0.9
    # )
    # print("\n[Q5 | Graph, k=0.9] 结果：")
    # for k_, v in ans_k.items():
    #     print(" ", k_, ":", v)

    # 3) L1 精修 + 间隔惩罚更强：k=0.7
    ans_k_L1 = solve_q5_graph(
        dt_mask=0.02,              # L1 计算更重，步长可略放大
        budgets_per_uav=(3,3,3,3,3),
        min_drop_gap=1.0,
        do_gap_fill=True,
        do_polish_L1=True,         # 开启 L1 精修（圆柱表面采样）
        N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
        k=0.9
    )
    print("\n[Q5 | Graph + L1 polish, k=0.9] 结果：")
    for k_, v in ans_k_L1.items():
        print(" ", k_, ":", v)
