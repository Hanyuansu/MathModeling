# -*- coding: utf-8 -*-
"""
CUMCM 2025 A —— Q5（分组法·只看覆盖·每机必须3枚）
修正版要点：
1) 强制三枚导弹都有 > min_cover_s_each 的覆盖（可调）
2) 即使 min_drop_gap=0 也避免同一 UAV 选到完全相同 (t_drop, tau)
3) 未达标导弹优先，必要时在其最大空档处“合成一条”补齐

作者：ChatGPT
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# =============== 场景常数 ===============
g = 9.81
VM = 300.0
V_SINK = 3.0
R_SMOKE = 10.0
T_EFFECT = 20.0

R_TAR, H_TAR = 7.0, 10.0
CYL_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)
P_TARGET   = np.array([0.0, 200.0, 5.0], dtype=float)

MISSILES = [
    {"name": "M1", "M0": np.array([20000.0,    0.0, 2000.0], dtype=float)},
    {"name": "M2", "M0": np.array([19000.0,  600.0, 2100.0], dtype=float)},
    {"name": "M3", "M0": np.array([18000.0, -600.0, 1900.0], dtype=float)},
]
UAVS = [
    {"name": "FY1", "U0": np.array([17800.0,     0.0, 1800.0], dtype=float)},
    {"name": "FY2", "U0": np.array([12000.0,  1400.0, 1400.0], dtype=float)},
    {"name": "FY3", "U0": np.array([ 6000.0, -3000.0,  700.0], dtype=float)},
    {"name": "FY4", "U0": np.array([11000.0,  2000.0, 1800.0], dtype=float)},
    {"name": "FY5", "U0": np.array([13000.0, -2000.0, 1300.0], dtype=float)},
]

# =============== 物理/几何 ===============
def missile_hit_time(m0: np.ndarray) -> float:
    return float(np.linalg.norm(m0) / VM)

def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v));  return v/n if n>0 else v

def missile_pos(m0: np.ndarray, t: float) -> np.ndarray:
    d = unit(-m0);  return m0 + VM*d*t

def uav_pos(u0: np.ndarray, theta: float, v_u: float, t: float) -> np.ndarray:
    hx, hy = math.cos(theta), math.sin(theta)
    return np.array([u0[0]+v_u*hx*t, u0[1]+v_u*hy*t, u0[2]], dtype=float)

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
    a = float(np.dot(X-P, v)/vv); a = 0.0 if a<0 else (1.0 if a>1 else a)
    Y = P + a*v
    return float(np.linalg.norm(X - Y))

def covered_L0_at_time(m0, p_target, s_burst, t_burst, t) -> bool:
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    return (point_to_segment_dist(p_target, m_t, s_t) <= R_SMOKE)

def clip(x, lo, hi): return lo if x<lo else (hi if x>hi else x)

# =============== L1 采样（如需可开） ===============
def cyl_points_top_bottom(N_ang: int = 36) -> np.ndarray:
    cx, cy, _ = CYL_CENTER; out=[]
    for z in (0.0, H_TAR):
        for k in range(N_ang):
            ang = 2.0*math.pi*k/N_ang
            out.append((cx+R_TAR*math.cos(ang), cy+R_TAR*math.sin(ang), z))
    return np.array(out, dtype=float)

def cyl_points_side(N_ang: int = 36, N_z: int = 7) -> np.ndarray:
    cx, cy, _ = CYL_CENTER; zs = np.linspace(0.0, H_TAR, N_z); out=[]
    for z in zs:
        for k in range(N_ang):
            ang = 2.0*math.pi*k/N_ang
            out.append((cx+R_TAR*math.cos(ang), cy+R_TAR*math.sin(ang), z))
    return np.array(out, dtype=float)

def build_cylinder_samples(N_ang=36, N_Z=7, include_side=True) -> np.ndarray:
    pts=[cyl_points_top_bottom(N_ang)]
    if include_side: pts.append(cyl_points_side(N_ang, N_Z))
    return np.concatenate(pts, axis=0)

def covered_L1_at_time_vectorized(m0: np.ndarray, s_burst: np.ndarray, t_burst: float, t: float, PTS: np.ndarray) -> bool:
    m_t = missile_pos(m0, t); s_t = smoke_center_after_burst(s_burst, t, t_burst)
    v = m_t - PTS;  w = s_t - PTS
    vv = np.sum(v*v, axis=1)
    alpha = np.divide(np.sum(w*v, axis=1), vv, out=np.zeros_like(vv), where=vv>0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    Y = PTS + alpha[:, None]*v
    dist = np.linalg.norm(s_t - Y, axis=1)
    return bool(np.any(dist <= R_SMOKE))

# =============== 时间网格 & 掩码 ===============
def _time_grid(T_hit: float, dt: float) -> np.ndarray:
    return np.arange(0.0, T_hit + 1e-12, dt)

def _mask_for_candidate(uav_idx: int, theta: float, v: float, t_drop: float, tau: float,
                        tgrids: Dict[str,np.ndarray],
                        mode: str = "L0", PTS: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    只看覆盖（不惩罚间隔）；返回三枚导弹的覆盖布尔掩码
    """
    masks={}
    u0 = UAVS[uav_idx]["U0"]
    t_burst = t_drop + tau
    s_burst = burst_point(u0, theta, v, t_drop, tau)
    for m in MISSILES:
        name, m0 = m["name"], m["M0"]
        T_HIT = missile_hit_time(m0)
        tgrid = tgrids[name]
        mask = np.zeros_like(tgrid, dtype=bool)
        if (s_burst[2] > 0.0) and (t_burst < T_HIT):
            t_start, t_end = t_burst, min(t_burst + T_EFFECT, T_HIT)
            idx = np.where((tgrid >= t_start) & (tgrid <= t_end))[0]
            for k in idx:
                t = float(tgrid[k])
                ok = covered_L0_at_time(m0, P_TARGET, s_burst, t_burst, t) if mode=="L0" \
                     else covered_L1_at_time_vectorized(m0, s_burst, t_burst, t, PTS)
                if ok: mask[k]=True
        masks[name]=mask
    return masks

def _score_union(union_masks: Dict[str,np.ndarray], dt: float) -> float:
    return sum(float(union_masks[name].sum()*dt) for name in union_masks)

def _marginal_gain(cand_masks: Dict[str,np.ndarray], union_masks: Dict[str,np.ndarray], dt: float) -> float:
    gain=0.0
    for nm in union_masks:
        old = union_masks[nm]
        new = np.logical_or(old, cand_masks[nm])
        gain += float((new.sum()-old.sum())*dt)
    return gain

# =============== 锚点反解（生成 θ、v、t_drop、tau 候选） ===============
def _candidate_from_anchor(u0: np.ndarray, m0: np.ndarray,
                           frac: float, alpha: float, tau_mult: float,
                           clamp_eps: float = 0.10) -> Tuple[float,float,float,float]:
    """
    frac 决定 t_burst 的早晚，alpha 决定“预瞄点”在 P_TARGET→missile(t_b) 之间的插值系数，
    tau_mult 作为下落时间基准的倍率
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
    tau_base = math.sqrt(max(0.0, 2.0*(u0z - Yz))/g) if u0z > Yz else 0.2
    tau_max_by_t = max(0.2, t_b - clamp_eps)
    tau = clip(tau_base * tau_mult, 0.2, min(12.0, tau_max_by_t))
    t_drop = t_b - tau
    return theta%(2.0*math.pi), v, t_drop, tau

# =============== 固定航向上的投放条目（预计算） ===============
def candidates_on_locked_course(uav_idx: int, theta: float, v: float,
                                tgrids: Dict[str,np.ndarray], dt_mask: float,
                                mode: str="L0", PTS: Optional[np.ndarray]=None,
                                t_burst_grid: Optional[np.ndarray]=None,
                                tau_list=(0.6,1.0,1.6,2.4,3.6,5.5,8.0,11.0)) -> List[Dict[str,Any]]:
    """
    已锁定 (θ,v) 后，枚举 (t_burst, tau) 生成候选；只保留有覆盖的。
    """
    u0 = UAVS[uav_idx]["U0"]
    T_min_hit = min(missile_hit_time(m["M0"]) for m in MISSILES)
    if t_burst_grid is None:
        t_end = max(3.0, min(59.0, T_min_hit-0.5))
        t_burst_grid = np.arange(3.0, t_end+1e-9, 1.6)

    cands=[]
    for t_b in t_burst_grid:
        for tau in tau_list:
            t_drop = t_b - tau
            if t_drop < 0.0 or t_drop > 60.0:  # 投放窗口
                continue
            sb = burst_point(u0, theta, v, t_drop, tau)
            if sb[2] <= 0.0:
                continue
            masks = _mask_for_candidate(uav_idx, theta, v, t_drop, tau, tgrids, mode=mode, PTS=PTS)
            cover_sum = sum(float(masks[name].sum()*dt_mask) for name in tgrids)
            if cover_sum <= 0.0: continue
            cands.append({
                "uav": uav_idx, "theta": theta, "v": v,
                "t_drop": t_drop, "tau": tau, "t_burst": t_b,
                "mask_by_missile": masks, "score_cover": cover_sum
            })
    # 轻去重：同一 t_burst 只留分数高的
    cands.sort(key=lambda c:c["t_burst"])
    filtered=[]
    for c in cands:
        if not filtered or abs(c["t_burst"] - filtered[-1]["t_burst"]) > 0.10:
            filtered.append(c)
        else:
            if c["score_cover"] > filtered[-1]["score_cover"]:
                filtered[-1]=c
    filtered.sort(key=lambda c:c["score_cover"], reverse=True)
    return filtered

# =============== 单架 UAV 的 (θ,v) 小候选库 ===============
def build_heading_bank_per_uav(uav_idx: int, tgrids: Dict[str,np.ndarray], dt_mask: float,
                               per_uav_keep: int = 6) -> List[Tuple[float,float,float]]:
    """
    返回 [(theta, v, local_score)]*K；local_score=该航向上“该 UAV 取 3 枚”的并集覆盖
    """
    u0 = UAVS[uav_idx]["U0"]
    fracs = (0.10,0.18,0.25,0.40,0.55,0.70,0.85,0.92,0.96)
    alphas= (0.60,0.70,0.80,0.88,0.92,0.96)
    taus  = (0.70,1.00,1.30)
    raw=[]
    for m in MISSILES:
        m0=m["M0"]
        for f in fracs:
            for a in alphas:
                for tm in taus:
                    th, v, td, ta = _candidate_from_anchor(u0, m0, f, a, tm)
                    raw.append((th, v))
    # 去重（θ±2°、v±2m/s）
    uniq=[]
    for th,v in raw:
        ok=True
        for (th2,v2,_) in uniq:
            d = abs((th-th2+math.pi)%(2*math.pi)-math.pi)
            if d <= math.radians(2.0) and abs(v-v2) <= 2.0:
                ok=False; break
        if ok: uniq.append([th,v,0.0])
    # 评估每组 (θ,v) 的“3 枚本地贪心并集得分”
    for i in range(len(uniq)):
        th,v,_ = uniq[i]
        cands = candidates_on_locked_course(uav_idx, th, v, tgrids, dt_mask, mode="L0")
        # 本地取3枚（只看覆盖）
        union = {nm: np.zeros_like(tgrids[nm], dtype=bool) for nm in tgrids}
        chosen=[]
        for _ in range(3):
            best=None; best_gain=-1e18
            for c in cands:
                if c in chosen: continue
                # 简单“非完全重复”过滤（阈值 0.02s）
                if any(abs(d["t_drop"]-c["t_drop"])<0.02 and abs(d["tau"]-c["tau"])<0.02 for d in chosen):
                    continue
                gain = _marginal_gain(c["mask_by_missile"], union, dt_mask)
                if gain>best_gain: best, best_gain = c, gain
            if best is None or best_gain<=0.0: break
            chosen.append(best)
            for nm in union:
                union[nm]=np.logical_or(union[nm], best["mask_by_missile"][nm])
        uniq[i][2]=_score_union(union, dt_mask)
    uniq.sort(key=lambda t:t[2], reverse=True)
    return uniq[:per_uav_keep]

# =============== 全局贪心（关键修正版） ===============
def greedy_select_locked(all_cands: List[Dict[str,Any]], tgrids: Dict[str,np.ndarray],
                         budgets_per_uav=(3,3,3,3,3), min_drop_gap: float = 0.0,
                         dt_mask: float=0.02,
                         heading_tol_deg: float = 2.0, speed_tol: float = 1.5,
                         require_all_missiles: bool=True, min_cover_s_each: float=0.2,
                         force_use_all: bool=True, boost_weight: float=1e6) -> Tuple[List[Dict[str,Any]], Dict[str,np.ndarray]]:
    """
    - 航向/速度锁定：某 UAV 第一次被选中后，后续必须同航向&速度（容差）
    - 未覆盖导弹优先：若某些导弹 < min_cover_s_each，则只考虑能给这些导弹带来正增益的候选（并用大权重）
    - 精确去重：同一 UAV 的 (t_drop, tau) 若接近相同（<0.02s），视为重复，禁止再次入选
    """
    budgets = {i:budgets_per_uav[i] for i in range(len(UAVS))}
    union = {nm: np.zeros_like(tgrids[nm], dtype=bool) for nm in tgrids}
    chosen: List[Dict[str,Any]]=[]
    remain = all_cands[:]
    total_budget = sum(budgets.values())
    tol_rad = math.radians(heading_tol_deg)
    locks: Dict[int, Optional[Tuple[float,float]]] = {i:None for i in range(len(UAVS))}
    # 已选 (t_drop,tau) 指纹，避免重复
    seen: Dict[int, List[Tuple[float,float]]] = {i:[] for i in range(len(UAVS))}

    def feasible(c):
        u=c["uav"]
        if budgets[u] <= 0: return False
        # 航向/速度锁
        thv = locks.get(u)
        if thv is not None:
            th_lock, v_lock = thv
            d = abs((c["theta"]-th_lock+math.pi)%(2*math.pi)-math.pi)
            if d > tol_rad: return False
            if abs(c["v"]-v_lock) > speed_tol: return False
        # 精确去重器（不考虑间隔，但不允许同值重复）
        for (td0,ta0) in seen[u]:
            if abs(td0-c["t_drop"])<0.02 and abs(ta0-c["tau"])<0.02:  # 2/100 秒阈值
                return False
        return True

    def cov_per_missile(U):
        return {nm: float(U[nm].sum()*dt_mask) for nm in U}

    while len(chosen) < total_budget:
        # 尚未达标的导弹集合
        need=set()
        if require_all_missiles:
            cov_now = cov_per_missile(union)
            need = {nm for nm,cv in cov_now.items() if cv < min_cover_s_each}

        best=None; best_score=-1e18
        for c in remain:
            if not feasible(c): continue
            # 计算边际增益
            global_gain = _marginal_gain(c["mask_by_missile"], union, dt_mask)
            if need:
                tg = 0.0
                for nm in need:
                    old = union[nm]
                    new = np.logical_or(old, c["mask_by_missile"][nm])
                    tg += float((new.sum()-old.sum())*dt_mask)
                if tg <= 0.0:  # 不能改善未覆盖导弹，直接跳过
                    continue
                score = boost_weight*tg + global_gain
            else:
                score = global_gain
            if score > best_score:
                best, best_score = c, score

        # 若当前阶段无候选可投，但还需要“用满”，则允许退化挑“单体得分最高但不重复”的
        if best is None:
            if not force_use_all: break
            tmp=None; tv=-1e18
            for c in remain:
                if not feasible(c): continue
                if c["score_cover"] > tv:
                    tv=c["score_cover"]; tmp=c
            best=tmp
            if best is None: break

        # 接受 best
        chosen.append(best)
        u = best["uav"]
        budgets[u] -= 1
        # 累并集
        for nm in union:
            union[nm] = np.logical_or(union[nm], best["mask_by_missile"][nm])
        # 记录“已使用的 (t_drop, tau)”
        seen[u].append((best["t_drop"], best["tau"]))
        # 首次锁定 (θ,v)
        if locks[u] is None:
            locks[u] = (best["theta"], best["v"])
        # 从候选集中移除该对象
        remain = [x for x in remain if x is not best]
        # 该 UAV 用尽则移除其余候选
        if budgets[u] <= 0:
            remain = [x for x in remain if x["uav"] != u]
        # 所有配额用尽
        if sum(max(0,b) for b in budgets.values()) == 0:
            break

    return chosen, union

# =============== 针对未覆盖导弹的“空档补齐” ===============
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

def _synthesize_for_gap(uav_idx: int, missile_idx: int, center_t: float,
                        dt_mask: float, tgrids: Dict[str, np.ndarray],
                        locked: Optional[Tuple[float,float]]) -> Optional[Dict[str,Any]]:
    """
    在 gap 中心附近“合成一条”候选（仅调时序），尽量给该导弹产生覆盖
    """
    m0 = MISSILES[missile_idx]["M0"]; u0 = UAVS[uav_idx]["U0"]
    T_HIT = missile_hit_time(m0)
    t_b = clip(center_t, 0.8, min(59.2, T_HIT-0.5))
    frac = t_b / min(60.0, T_HIT-2.0)
    alphas_try = (0.92, 0.96, 0.985)
    tau_mult_try = (0.70, 0.85, 1.00, 1.15)

    def mk(theta, v, td, ta):
        cand = {"uav": uav_idx, "theta":theta, "v":v, "t_drop":td, "tau":ta}
        masks = _mask_for_candidate(uav_idx, theta, v, td, ta, tgrids)
        cover_sum = sum(float(masks[nm].sum() * dt_mask) for nm in tgrids)
        cand.update({"mask_by_missile": masks, "score_cover": cover_sum, "t_burst": td+ta})
        return cand

    best=None
    if locked is not None:
        th, v = locked
        _,_,td0,ta0 = _candidate_from_anchor(u0, m0, frac, 0.96, 1.0)
        tdN  = [clip(td0 + d, 0.0, 60.0) for d in (-1.6,-1.0,-0.4,0.0,0.4,1.0,1.6)]
        for a in alphas_try:
            _,_,td,ta = _candidate_from_anchor(u0, m0, frac, a, 1.0)
            for td1 in tdN:
                for mul in tau_mult_try:
                    ta1 = clip(ta*mul, 0.2, min(12.0, t_b-0.1))
                    c1 = mk(th, v, td1, ta1)
                    if (best is None) or (c1["score_cover"] > best["score_cover"]):
                        best = c1
        return best

    th0,v0,td0,ta0 = _candidate_from_anchor(u0, m0, frac, 0.96, 1.0)
    yawN = [th0 + math.radians(d) for d in (-10,-6,-3,0,3,6,10)]
    vN   = [clip(v0 + dv, 70.0, 140.0) for dv in (-14,-8,0,8,14)]
    tdN  = [clip(td0 + d, 0.0, 60.0)   for d in (-1.6,-1.0,-0.4,0.0,0.4,1.0,1.6)]
    for a in alphas_try:
        th,v,td,ta = _candidate_from_anchor(u0, m0, frac, a, 1.0)
        for th1 in yawN:
            for v1 in vN:
                for td1 in tdN:
                    for mul in tau_mult_try:
                        ta1 = clip(ta*mul, 0.2, min(12.0, t_b-0.1))
                        c1 = mk(th1, v1, td1, ta1)
                        if (best is None) or (c1["score_cover"] > best["score_cover"]):
                            best = c1
    return best

def force_cover_every_missile(chosen: List[Dict[str,Any]], union_masks: Dict[str,np.ndarray],
                              tgrids: Dict[str,np.ndarray], dt_mask: float,
                              budgets_per_uav, min_cover_s_each: float = 0.2):
    """
    若存在导弹覆盖 < min_cover_s_each，则在其最大空档处择优“合成”补射直到达标或用满
    """
    budgets = {i:budgets_per_uav[i] for i in range(len(UAVS))}
    locks: Dict[int, Optional[Tuple[float,float]]] = {}
    for c in chosen:
        u=c["uav"]; budgets[u]-=1
        if u not in locks: locks[u]=(c["theta"], c["v"])

    for midx,_ in enumerate(MISSILES):
        name = MISSILES[midx]["name"]
        cov = float(union_masks[name].sum()*dt_mask)
        while cov < min_cover_s_each and sum(max(0,b) for b in budgets.values())>0:
            center_t,_ = _largest_gap(union_masks[name], tgrids[name])
            best=None; best_gain=-1e18
            for u in range(len(UAVS)):
                if budgets[u] <= 0: continue
                synth=_synthesize_for_gap(u, midx, center_t, dt_mask, tgrids, locks.get(u))
                if synth is None: continue
                gain=_marginal_gain(synth["mask_by_missile"], union_masks, dt_mask)
                if gain>best_gain:
                    best,best_gain=synth,gain
            if best is None: break
            chosen.append(best); budgets[best["uav"]]-=1
            if best["uav"] not in locks:
                locks[best["uav"]] = (best["theta"], best["v"])
            for nm in tgrids:
                union_masks[nm]=np.logical_or(union_masks[nm], best["mask_by_missile"][nm])
            cov = float(union_masks[name].sum()*dt_mask)
    return chosen, union_masks

# =============== 报表 ===============
def _drop_point(u0: np.ndarray, theta: float, v_u: float, t_drop: float) -> np.ndarray:
    return uav_pos(u0, theta, v_u, t_drop)

def build_report_rows(selected_raw: List[Dict[str, Any]],
                      tgrids: Dict[str, np.ndarray],
                      dt_mask: float) -> List[Dict[str, Any]]:
    # 为每架 UAV 内部按 t_drop 编号 FYi-1/2/3...
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
        theta_deg=(math.degrees(c["theta"])%360.0); v_u=c["v"]; t_drop=c["t_drop"]; tau=c["tau"]
        r_drop=_drop_point(u0, c["theta"], v_u, t_drop)
        s_burst=burst_point(u0, c["theta"], v_u, t_drop, tau)
        smoke_id=f"{u_name}-{c.get('_seq_in_uav',1)}"
        for m in MISSILES:
            name=m["name"]; mask=c["mask_by_missile"][name]
            eff_s=float(mask.sum())*dt_mask
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

# =============== 主流程 ===============
def solve_q5_use_all_only_coverage(
    # 判定
    mode: str = "L0",          # "L0" 或 "L1"
    dt_mask: float = 0.02,
    N_ANG:int=36, N_Z:int=7, INCLUDE_SIDE=True,
    # 候选与航向库
    per_uav_keep: int = 6,
    # 预算与锁定
    budgets_per_uav=(3,3,3,3,3),
    heading_tol_deg: float = 2.0, speed_tol: float = 1.5,
    # 覆盖要求与兜底
    require_all_missiles: bool = True,
    min_cover_s_each: float = 0.2,
    force_use_all: bool = True
) -> Dict[str,Any]:

    # 时间网格
    tgrids = {m["name"]: _time_grid(missile_hit_time(m["M0"]), dt_mask) for m in MISSILES}
    PTS = build_cylinder_samples(N_ang=N_ANG, N_Z=N_Z, include_side=INCLUDE_SIDE) if mode=="L1" else None

    # 每架 UAV 构建 (θ,v) 小库 + 预计算条目（缓存）
    banks=[]; cand_cache={}
    for u in range(5):
        bank_u = build_heading_bank_per_uav(u, tgrids, dt_mask, per_uav_keep=per_uav_keep)
        banks.append(bank_u)
        for i,(th,v,_) in enumerate(bank_u):
            cand_cache[(u,i)] = candidates_on_locked_course(
                u, th, v, tgrids, dt_mask, mode=("L1" if PTS is not None else "L0"), PTS=PTS
            )

    # 初始化：每架 UAV 取本地最优航向
    sel_idx=[0,0,0,0,0]
    # 合并全部候选，进入全局贪心
    all_cands=[]
    for u in range(5):
        all_cands += cand_cache[(u, sel_idx[u])]

    chosen, union = greedy_select_locked(
        all_cands, tgrids,
        budgets_per_uav=budgets_per_uav,
        min_drop_gap=0.0,                # 不考虑间隔
        dt_mask=dt_mask,
        heading_tol_deg=heading_tol_deg, speed_tol=speed_tol,
        require_all_missiles=require_all_missiles, min_cover_s_each=min_cover_s_each,
        force_use_all=force_use_all
    )

    # 若还有导弹未达标，再做一次“空档补齐”
    chosen, union = force_cover_every_missile(chosen, union, tgrids, dt_mask, budgets_per_uav, min_cover_s_each)

    # 统计 per-missile
    per_missile=[]
    for m in MISSILES:
        name=m["name"]; tgrid=tgrids[name]; mask=union[name]
        cov=float(mask.sum()*dt_mask)
        # 区间
        intervals=[]; in_seg=False; a=None
        for k in range(len(mask)):
            if mask[k] and not in_seg: in_seg=True; a=float(tgrid[k])
            if in_seg and (k==len(mask)-1 or (not mask[k+1])):
                b=float(tgrid[k]); intervals.append((round(a,3), round(b,3))); in_seg=False
        per_missile.append({"missile":name, "T_hit": round(missile_hit_time([mm for mm in MISSILES if mm["name"]==name][0]["M0"]),3),
                            "cover_s": round(cov,3), "intervals": intervals})

    rows = build_report_rows(chosen, tgrids, dt_mask)

    return {
        "method": "use_all_only_coverage (fixed)",
        "per_missile": per_missile,
        "cover_sum_s": round(sum(x["cover_s"] for x in per_missile), 3),
        "rows": rows,
        "config": {
            "mode": mode, "dt_mask": dt_mask,
            "per_uav_keep": per_uav_keep,
            "heading_tol_deg": heading_tol_deg, "speed_tol": speed_tol,
            "require_all_missiles": require_all_missiles, "min_cover_s_each": min_cover_s_each,
            "force_use_all": force_use_all
        }
    }

# =============== 运行示例（只打印） ===============
if __name__ == "__main__":
    ans = solve_q5_use_all_only_coverage(
        mode="L1",            # 需要更稳可改为 "L1"
        dt_mask=0.02,
        per_uav_keep=6,
        budgets_per_uav=(3,3,3,3,3),
        heading_tol_deg=2.0, speed_tol=1.5,
        require_all_missiles=True, min_cover_s_each=0.2,   # ★确保三导弹都有覆盖
        force_use_all=True
    )

    print("\n[Q5 | 结果摘要]")
    for x in ans["per_missile"]:
        print(" ", x)
    print("  覆盖总时长 (s)：", ans["cover_sum_s"])

    print("\n[Q5 | 报表] 无人机×烟幕投放×导弹干扰明细（12列）：")
    print_report_rows(ans["rows"])
