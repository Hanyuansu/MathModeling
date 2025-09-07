import math
import copy
import statistics as stats
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
rng = np.random.default_rng(2024)

g = 9.81
VM = 300.0
V_SINK = 3.0
R_SMOKE = 10.0
T_EFFECT = 20.0

DT_STEP = 0.015

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

BUDGETS = (3, 3, 3, 3, 3)
MIN_DROP_GAP = 1.0

HEADING_TOL_DEG = 2.0
SPEED_TOL       = 1.5

FRACS  = (0.10, 0.18, 0.25, 0.40, 0.55, 0.70, 0.85, 0.92, 0.96, 0.985)
ALPHAS = (0.60, 0.70, 0.80, 0.88, 0.92, 0.96, 0.985)
TAUS_MUL = (0.55, 0.70, 0.85, 1.00, 1.15, 1.30)
PER_UAV_KEEP = 36
DEDUP_EPS = 0.12

N_ANG, N_Z, INCLUDE_SIDE = 48, 9, True
POLISH_ROUNDS = 1

GLOBAL_SA_ON = True
SA_ITERS = 2000
SA_T0    = 0.5
SA_TEND  = 1e-3
SA_SIG_T = 0.6
SA_SIG_A = 0.30


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

def covered_L0_at_time(m0, p_target, s_burst, t_burst, t) -> bool:
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    return (point_to_segment_dist(p_target, m_t, s_t) <= R_SMOKE)

def cyl_points_top_bottom(N_ang: int = 48) -> np.ndarray:
    cx, cy, _ = CYL_CENTER; out=[]
    for z in (0.0, H_TAR):
        for k in range(N_ang):
            ang = 2.0*math.pi*k/N_ang
            out.append((cx+R_TAR*math.cos(ang), cy+R_TAR*math.sin(ang), z))
    return np.array(out, dtype=float)

def cyl_points_side(N_ang: int = 48, N_Z: int = 9) -> np.ndarray:
    cx, cy, _ = CYL_CENTER; zs = np.linspace(0.0, H_TAR, N_Z); out=[]
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

def _angdiff(a: float, b: float) -> float:
    d = abs((a - b + math.pi) % (2*math.pi) - math.pi)
    return d

def _feasible_with(cand, chosen, budgets, min_drop_gap,
                   locks: Dict[int, Optional[Tuple[float,float]]],
                   heading_tol_rad: float, speed_tol: float) -> bool:
    u = cand["uav"]
    if budgets[u] <= 0: return False
    for c in chosen:
        if c["uav"]==u and abs(float(c["t_drop"]) - float(cand["t_drop"])) < min_drop_gap:
            return False
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

def _force_fill_to_15(chosen_list, tgrids_loc, dt_loc):
    locks_tmp = {}
    for c in chosen_list:
        u = c["uav"]
        if u not in locks_tmp:
            locks_tmp[u] = (c["theta"], c["v"])
    for u in range(len(UAVS)):
        if u not in locks_tmp:
            u0 = UAVS[u]["U0"]
            th = math.atan2(P_TARGET[1]-u0[1], P_TARGET[0]-u0[0])
            locks_tmp[u] = (th, 100.0)

    used_per_uav = {i:0 for i in range(len(UAVS))}
    for c in chosen_list:
        used_per_uav[c["uav"]] += 1

    def ok_tdrop(u, td):
        for x in chosen_list:
            if x["uav"] == u and abs(float(x["t_drop"]) - float(td)) < MIN_DROP_GAP:
                return False
        return True

    for u in range(len(UAVS)):
        th, v = locks_tmp[u]
        pool_u = augment_locked_same_course(u, th, v, tgrids_loc, dt_loc, t_step=1.0)
        k = 0
        while used_per_uav[u] < BUDGETS[u]:
            pick = None
            while k < len(pool_u):
                if ok_tdrop(u, pool_u[k]["t_drop"]):
                    pick = pool_u[k]; k += 1; break
                k += 1
            if pick is None:
                td = 0.0
                while td <= 60.0 and (not ok_tdrop(u, td)):
                    td += 0.25
                tau = 0.3
                c = {"uav": u, "theta": th, "v": v, "t_drop": td, "tau": tau, "t_burst": td+tau}
                c["mask_by_missile"] = _mask_for_candidate_L0(u, c, tgrids_loc)
                c["score_cover"] = _cover_sum(c["mask_by_missile"], dt_loc)
                pick = c
            chosen_list.append(pick)
            used_per_uav[u] += 1
    return chosen_list

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

    def ok_gap_with_others(u, td, others):
        for x in others:
            if x["uav"]==u and abs(float(x["t_drop"]) - float(td)) < MIN_DROP_GAP:
                return False
        return True

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
                if not ok_gap_with_others(c["uav"], td, base):
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

def global_sa_optimize_L1(selected: List[Dict[str,Any]],
                          tgrids: Dict[str,np.ndarray],
                          dt_mask: float = DT_STEP,
                          iters: int = SA_ITERS,
                          T0: float = SA_T0,
                          Tend: float = SA_TEND,
                          sig_t: float = SA_SIG_T,
                          sig_tau_rel: float = SA_SIG_A,
                          N_ang:int=48, N_Z:int=9, INCLUDE_SIDE:bool=True):

    PTS = build_cylinder_samples(N_ang, N_Z, INCLUDE_SIDE)

    def rebuild(c):
        c2 = dict(c)
        c2["mask_by_missile"] = _mask_for_candidate_L1(c["uav"], c2, tgrids, PTS)
        c2["score_cover"] = _cover_sum(c2["mask_by_missile"], dt_mask)
        return c2

    def score_of(sol):
        union = {nm: np.zeros_like(tgrids[nm], dtype=bool) for nm in tgrids}
        for c in sol:
            for nm in tgrids:
                union[nm] = np.logical_or(union[nm], c["mask_by_missile"][nm])
        sc = sum(float(union[nm].sum() * dt_mask) for nm in tgrids)
        return sc, union

    def ok_gap_for_idx(sol, idx, td_new):
        u = sol[idx]["uav"]
        for j,c in enumerate(sol):
            if j==idx: continue
            if c["uav"]==u and abs(float(c["t_drop"]) - float(td_new)) < MIN_DROP_GAP:
                return False
        return True

    cur = [rebuild(c) for c in selected]
    cur_score, cur_union = score_of(cur)
    best = [dict(c) for c in cur]
    best_score = cur_score

    def snap(x):
        k = round(x / DT_STEP)
        return float(max(0.0, min(60.0, k * DT_STEP)))

    for k in range(iters):
        i = rng.integers(0, len(cur))
        c0 = cur[i]
        td0, ta0 = float(c0["t_drop"]), float(c0["tau"])

        trial = dict(c0)
        for _attempt in range(12):
            td = snap(td0 + rng.normal(0.0, sig_t))
            if not ok_gap_for_idx(cur, i, td):
                continue
            ta = float(clip(ta0 * (1.0 + rng.normal(0.0, sig_tau_rel)), 0.2, 12.0))
            trial["t_drop"] = td
            trial["tau"] = ta
            trial["t_burst"] = td + ta
            trial = rebuild(trial)
            break
        else:
            continue

        new_sol = cur[:i] + [trial] + cur[i+1:]
        new_score, _ = score_of(new_sol)

        T = T0 * (Tend / T0) ** (k / max(1, iters-1))
        if new_score >= cur_score or rng.random() < math.exp((new_score - cur_score) / max(1e-12, T)):
            cur = new_sol
            cur_score = new_score
            if cur_score > best_score:
                best = [dict(x) for x in cur]
                best_score = cur_score

    final_union = {nm: np.zeros_like(tgrids[nm], dtype=bool) for nm in tgrids}
    for c in best:
        for nm in tgrids:
            final_union[nm] = np.logical_or(final_union[nm], c["mask_by_missile"][nm])
    return best, final_union


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
            if eff_s<=0.0: continue
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

def solve_q5(
    dt_step: float = DT_STEP,
    do_polish_L1: bool = True,
    do_global_SA: bool = GLOBAL_SA_ON,
    heading_tol_deg: float = HEADING_TOL_DEG,
    speed_tol: float = SPEED_TOL
) -> Dict[str,Any]:
    candidates_L0, tgrids_L0 = build_candidates_L0(dt=dt_step)
    counts = {i: sum(1 for c in candidates_L0 if c["uav"]==i) for i in range(len(UAVS))}
    print("[debug] L0 per-UAV nonzero candidates:", counts)

    seed_chosen, seed_union, budgets, locks = seed_cover_all_missiles(
        candidates_L0, tgrids_L0, dt_step,
        budgets_per_uav=BUDGETS,
        min_drop_gap=MIN_DROP_GAP,
        heading_tol_deg=heading_tol_deg, speed_tol=speed_tol
    )
    chosen, union, budgets, locks = greedy_fill_after_seed(
        candidates_L0, tgrids_L0, dt_step,
        seed_chosen, seed_union, budgets, locks,
        budgets_per_uav=BUDGETS,
        min_drop_gap=MIN_DROP_GAP,
        heading_tol_deg=heading_tol_deg, speed_tol=speed_tol
    )
    chosen = _force_fill_to_15(chosen, tgrids_L0, dt_step)
    assert len(chosen) == sum(BUDGETS), f"[fatal] 只选了 {len(chosen)}/{sum(BUDGETS)}"

    mode = "L0 only"
    tgrids = {m["name"]: _time_grid(missile_hit_time(m["M0"]), dt_step) for m in MISSILES}
    if do_polish_L1:
        chosen, union = polish_L1_keep_course(chosen, tgrids, dt_mask=dt_step,
                                              rounds=POLISH_ROUNDS, N_ang=N_ANG, N_Z=N_Z, INCLUDE_SIDE=INCLUDE_SIDE)
        mode = "L0 → L1(polish)"

    if do_global_SA:
        before_score = sum(float(union[nm].sum()*dt_step) for nm in union)
        chosen, union = global_sa_optimize_L1(
            selected=chosen, tgrids=tgrids, dt_mask=dt_step,
            iters=SA_ITERS, T0=SA_T0, Tend=SA_TEND,
            sig_t=SA_SIG_T, sig_tau_rel=SA_SIG_A,
            N_ang=N_ANG, N_Z=N_Z, INCLUDE_SIDE=INCLUDE_SIDE
        )
        after_score = sum(float(union[nm].sum()*dt_step) for nm in union)
        print(f"[info] SA 提升覆盖: {before_score:.3f} → {after_score:.3f} 秒")
        mode += " → L1(SA)"

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

    return {
        "method": "graph_Q5",
        "mode": mode,
        "selected": out_sel,
        "per_missile": per_missile,
        "cover_sum_s": cover_sum,
        "used_total": len(chosen),
        "config": {
            "dt_step": dt_step,
            "budgets": BUDGETS,
            "min_drop_gap": MIN_DROP_GAP,
            "heading_tol_deg": HEADING_TOL_DEG, "speed_tol": SPEED_TOL,
            "L1_polish": do_polish_L1,
            "L1_SA": do_global_SA,
            "SA_iters": SA_ITERS,
            "N_ANG": N_ANG, "N_Z": N_Z, "INCLUDE_SIDE": INCLUDE_SIDE
        },
        "rows": rows
    }

def _name_to_idx():
    return {UAVS[i]["name"]: i for i in range(len(UAVS))}

def _name2idx():  # 兼容旧名
    return _name_to_idx()

def _ans_to_internal_selected(ans_selected):
    n2i = _name2idx()
    internal=[]
    for r in ans_selected:
        internal.append({
            "uav": n2i[str(r["uav"])],
            "theta": math.radians(float(r["theta_deg"])),
            "v": float(r["v_u_mps"]),
            "t_drop": float(r["t_drop"]),
            "tau": float(r["tau"]),
            "t_burst": float(r["t_drop"])+float(r["tau"])
        })
    return internal

def _recompute_cover_L1(selected_internal, dt: float, N_ANG:int=48, N_Z:int=9, INCLUDE_SIDE:bool=True):
    tgrids = {m["name"]: _time_grid(missile_hit_time(m["M0"]), dt) for m in MISSILES}
    union  = {nm: np.zeros_like(tgrids[nm], dtype=bool) for nm in tgrids}
    PTS    = build_cylinder_samples(N_ang=N_ANG, N_Z=N_Z, include_side=INCLUDE_SIDE)

    def rebuild_masks_L1(c):
        c2 = dict(c)
        c2["mask_by_missile"] = _mask_for_candidate_L1(c2["uav"], c2, tgrids, PTS)
        return c2

    for c in selected_internal:
        cc = rebuild_masks_L1(c)
        for nm in tgrids:
            union[nm] = np.logical_or(union[nm], cc["mask_by_missile"][nm])

    cover_sum = float(sum(union[nm].sum() * dt for nm in union))
    return cover_sum, union, tgrids

def _repair_min_gap_per_uav(selected_internal, min_gap: float = MIN_DROP_GAP):
    by_uav = {}
    for c in selected_internal:
        by_uav.setdefault(c["uav"], []).append(c)
    for u,lst in by_uav.items():
        lst.sort(key=lambda x: x["t_drop"])
        last = -1e9
        for c in lst:
            td = float(c["t_drop"])
            if td < last + min_gap:
                td = last + min_gap
                td = clip(td, 0.0, 60.0)
                c["t_drop"] = td
                c["t_burst"] = td + float(c["tau"])
            last = float(c["t_drop"])
    return selected_internal

def _perturb_solution(selected_internal, sigma_t: float = 0.6, sigma_tau_rel: float = 0.3, seed: int = None):
    rng_loc = np.random.default_rng(seed)
    out=[]
    for c in selected_internal:
        td = float(c["t_drop"]) + float(rng_loc.normal(0.0, sigma_t))
        ta = float(c["tau"])    * (1.0 + float(rng_loc.normal(0.0, sigma_tau_rel)))
        td = clip(td, 0.0, 60.0)
        ta = clip(ta, 0.2, 12.0)
        out.append({
            "uav": c["uav"], "theta": float(c["theta"]), "v": float(c["v"]),
            "t_drop": td, "tau": ta, "t_burst": td+ta
        })
    return _repair_min_gap_per_uav(out, MIN_DROP_GAP)

def _random_solution_global_like(selected_internal, seed: int = None):
    rng_loc = np.random.default_rng(seed)
    thv = {}
    for c in selected_internal:
        thv[c["uav"]] = (float(c["theta"]), float(c["v"]))
    cnt_per_uav = {i:0 for i in range(len(UAVS))}
    out=[]
    for u in range(len(UAVS)):
        th, v = thv[u]
        K = BUDGETS[u]
        tds = list(rng_loc.uniform(0.0, 60.0, size=K))
        tds.sort()
        for i in range(1, K):
            if tds[i] < tds[i-1] + MIN_DROP_GAP:
                tds[i] = min(60.0, tds[i-1] + MIN_DROP_GAP)
        for i in range(K):
            td  = float(tds[i])
            tau = float(rng_loc.uniform(0.2, 12.0))
            out.append({"uav": u, "theta": th, "v": v, "t_drop": td, "tau": tau, "t_burst": td+tau})
    return out


def monte_carlo_compare(ans, n_trials: int = 100,
                        sigma_t: float = 0.6, sigma_tau_rel: float = 0.3,
                        dt: float = DT_STEP, N_ANG: int = 48, N_Z: int = 9, INCLUDE_SIDE: bool = True,
                        also_global_random: bool = True):
    base_sel = _ans_to_internal_selected(ans["selected"])
    base_cov, _, _ = _recompute_cover_L1(base_sel, dt, N_ANG, N_Z, INCLUDE_SIDE)

    better_local = 0
    cov_local = []
    for i in range(n_trials):
        cand = _perturb_solution(base_sel, sigma_t=sigma_t, sigma_tau_rel=sigma_tau_rel, seed=2024+i)
        cov, _, _ = _recompute_cover_L1(cand, dt, N_ANG, N_Z, INCLUDE_SIDE)
        cov_local.append(cov)
        if cov > base_cov + 1e-9:
            better_local += 1

    print("\n[MC-Local] 局部扰动对比:")
    print(f"  基线覆盖(L1): {base_cov:.3f} s")
    print(f"  {n_trials} 组扰动 覆盖均值/中位/最大/最小: {stats.mean(cov_local):.3f} / {stats.median(cov_local):.3f} / {max(cov_local):.3f} / {min(cov_local):.3f}")
    print(f"  超过基线的比例: {better_local}/{n_trials} = {better_local/n_trials:.2%}")

    cov_glob = None
    if also_global_random:
        better_glob = 0
        cov_glob = []
        for i in range(n_trials):
            cand = _random_solution_global_like(base_sel, seed=4096+i)
            cov, _, _ = _recompute_cover_L1(cand, dt, N_ANG, N_Z, INCLUDE_SIDE)
            cov_glob.append(cov)
            if cov > base_cov + 1e-9:
                better_glob += 1
        print("\n[MC-Global] 全局随机对比（锁(θ,v)）:")
        print(f"  {n_trials} 组随机 覆盖均值/中位/最大/最小: {stats.mean(cov_glob):.3f} / {stats.median(cov_glob):.3f} / {max(cov_glob):.3f} / {min(cov_glob):.3f}")
        print(f"  超过基线的比例: {better_glob}/{n_trials} = {better_glob/n_trials:.2%}")

    return {
        "base": base_cov,
        "local": cov_local,
        "global": cov_glob
    }


def sweep_dt(ans, dt_list=(0.010, 0.015, 0.020, 0.030), N_ANG:int=48, N_Z:int=9, INCLUDE_SIDE:bool=True):
    sel = _ans_to_internal_selected(ans["selected"])
    out=[]
    for dt in dt_list:
        cov, _, _ = _recompute_cover_L1(sel, dt, N_ANG, N_Z, INCLUDE_SIDE)
        out.append((dt, cov))
    base = out[0][1]
    print("\n[稳健性] 步长灵敏度（L1）：")
    for dt, cov in out:
        print(f"  dt={dt:.3f}  cover={cov:.3f}  相对偏差={(cov-base)/max(1e-12,base):+.2%}")
    return out

def sweep_sampling(ans, NANG_list=(24,48,96), NZ_list=(5,9,13), dt: float = DT_STEP):
    sel = _ans_to_internal_selected(ans["selected"])
    base_cov, _, _ = _recompute_cover_L1(sel, dt, N_ANG=48, N_Z=9, INCLUDE_SIDE=True)
    print("\n[稳健性] 采样密度灵敏度（L1）：")
    print(f"  基线(N_ANG=48,N_Z=9): {base_cov:.3f} s")
    out=[]
    for na in NANG_list:
        for nz in NZ_list:
            cov, _, _ = _recompute_cover_L1(sel, dt, N_ANG=na, N_Z=nz, INCLUDE_SIDE=True)
            out.append((na, nz, cov))
            print(f"  N_ANG={na:>3}, N_Z={nz:>2} -> cover={cov:.3f}  偏差={(cov-base_cov)/max(1e-12,base_cov):+.2%}")
    return out

def perturbation_curve(ans, sig_t_grid=(0.0,0.2,0.4,0.6,0.8,1.0), sig_tau_rel: float = 0.30,
                       reps: int = 30, dt: float = DT_STEP, N_ANG:int=48, N_Z:int=9):
    sel = _ans_to_internal_selected(ans["selected"])
    base_cov, _, _ = _recompute_cover_L1(sel, dt, N_ANG, N_Z, True)
    print("\n[扰动曲线] 覆盖-噪声关系（σ_t, 固定 σ_tau_rel=%.2f）:" % sig_tau_rel)
    print(f"  基线覆盖: {base_cov:.3f} s")
    table=[]
    for st in sig_t_grid:
        vals=[]
        for r in range(reps):
            cand = _perturb_solution(sel, sigma_t=st, sigma_tau_rel=sig_tau_rel, seed=7000+r)
            cov, _, _ = _recompute_cover_L1(cand, dt, N_ANG, N_Z, True)
            vals.append(cov)
        mean_v = float(np.mean(vals))
        p10   = float(np.percentile(vals, 10))
        p50   = float(np.percentile(vals, 50))
        p90   = float(np.percentile(vals, 90))
        print(f"  σ_t={st:.2f} -> mean={mean_v:.3f}, P10={p10:.3f}, P50={p50:.3f}, P90={p90:.3f}")
        table.append((st, mean_v, p10, p50, p90))
    return table

def validate_q5(ans,
                n_trials: int = 100,
                local_sigma_t: float = 0.6,
                local_sigma_tau_rel: float = 0.30,
                dt_list=(0.010,0.015,0.020,0.030),
                NANG_list=(24,48,96),
                NZ_list=(5,9,13),
                curve_sig_t=(0.0,0.2,0.4,0.6,0.8,1.0),
                curve_reps: int = 30):
    _ = monte_carlo_compare(ans, n_trials=n_trials,
                            sigma_t=local_sigma_t, sigma_tau_rel=local_sigma_tau_rel,
                            dt=DT_STEP, N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
                            also_global_random=True)
    _ = sweep_dt(ans, dt_list=dt_list, N_ANG=48, N_Z=9, INCLUDE_SIDE=True)
    _ = sweep_sampling(ans, NANG_list=NANG_list, NZ_list=NZ_list, dt=DT_STEP)
    _ = perturbation_curve(ans, sig_t_grid=curve_sig_t, sig_tau_rel=local_sigma_tau_rel,
                           reps=curve_reps, dt=DT_STEP, N_ANG=48, N_Z=9)

# 邻域检验
def _group_by_uav(selected_internal):
    by={}
    for c in selected_internal:
        by.setdefault(c["uav"], []).append(c)
    for u,lst in by.items():
        lst.sort(key=lambda x: x["t_drop"])
    return by

def _apply_uniform_shift_per_uav(base_sel, eps_t: float, eps_tau: float, rng_loc):

    cand = copy.deepcopy(base_sel)
    by = _group_by_uav(cand)
    for u,lst in by.items():
        dt_u = float(rng_loc.uniform(-eps_t,  eps_t))
        da_u = float(rng_loc.uniform(-eps_tau, eps_tau))
        for c in lst:
            td = clip(float(c["t_drop"]) + dt_u, 0.0, 60.0)
            ta = clip(float(c["tau"])    + da_u, 0.2, 12.0)
            c["t_drop"]  = td
            c["tau"]     = ta
            c["t_burst"] = td + ta
    return cand

def _apply_small_jitter_per_shot(base_sel, eps_t: float, eps_tau: float, rng_loc):
    cand = copy.deepcopy(base_sel)
    by = _group_by_uav(cand)
    for u,lst in by.items():
        for c in lst:
            td = clip(float(c["t_drop"]) + float(rng_loc.uniform(-eps_t,  eps_t)), 0.0, 60.0)
            ta = clip(float(c["tau"])    + float(rng_loc.uniform(-eps_tau, eps_tau)), 0.2, 12.0)
            c["t_drop"]  = td
            c["tau"]     = ta
            c["t_burst"] = td + ta
        lst.sort(key=lambda x:x["t_drop"])
        for i in range(1, len(lst)):
            if lst[i]["t_drop"] < lst[i-1]["t_drop"] + MIN_DROP_GAP:
                lst[i]["t_drop"] = min(60.0, lst[i-1]["t_drop"] + MIN_DROP_GAP)
        if lst and lst[-1]["t_drop"] > 60.0:
            lst[-1]["t_drop"] = 60.0
        for i in reversed(range(len(lst)-1)):
            hi = lst[i+1]["t_drop"] - MIN_DROP_GAP
            if lst[i]["t_drop"] > hi:
                lst[i]["t_drop"] = max(0.0, hi)
        for c in lst:
            c["t_burst"] = float(c["t_drop"]) + float(c["tau"])
    return cand

def neighborhood_validate_best(ans,
                               n_samples: int = 100,
                               eps_t: float = 0.30,
                               eps_tau: float = 0.20,
                               ratio_per_shot_jitter: float = 0.25,
                               dt: float = DT_STEP,
                               N_ANG: int = 48, N_Z: int = 9, INCLUDE_SIDE: bool = True,
                               seed: int = 20250):
    rng_loc = np.random.default_rng(seed)
    base_sel = _ans_to_internal_selected(ans["selected"])
    base_cov, _, _ = _recompute_cover_L1(base_sel, dt, N_ANG, N_Z, INCLUDE_SIDE)

    covers = []
    better = 0
    best_improve = (-1e9, None)
    for k in range(n_samples):
        if rng_loc.random() < ratio_per_shot_jitter:
            cand = _apply_small_jitter_per_shot(base_sel, eps_t, eps_tau, rng_loc)
        else:
            cand = _apply_uniform_shift_per_uav(base_sel, eps_t, eps_tau, rng_loc)
        cov, _, _ = _recompute_cover_L1(cand, dt, N_ANG, N_Z, INCLUDE_SIDE)
        covers.append(cov)
        if cov > base_cov + 1e-9:
            better += 1
        if cov > best_improve[0]:
            best_improve = (cov, k)

    print("\n[邻域检验（最优解周围）]")
    print(f"  基线覆盖(L1): {base_cov:.3f} s")
    print(f"  邻域半径: eps_t={eps_t:.2f}s, eps_tau={eps_tau:.2f}s, 样本数={n_samples}")
    print(f"  覆盖统计: mean={np.mean(covers):.3f}, median={np.median(covers):.3f}, "
          f"max={np.max(covers):.3f}, min={np.min(covers):.3f}")
    print(f"  超过基线的比例: {better}/{n_samples} = {better/n_samples:.2%}")
    if best_improve[0] > base_cov:
        print(f"  最佳样本提升: +{(best_improve[0]-base_cov):.3f} s (绝对覆盖 {best_improve[0]:.3f})")
    else:
        print("  邻域内未发现比基线更好的样本。")
    return {
        "base": base_cov,
        "covers": covers,
        "better_count": better,
        "best_cov": best_improve[0]
    }

if __name__ == "__main__":
    ans = solve_q5(
        dt_step=DT_STEP,
        do_polish_L1=True,
        do_global_SA=True,
        heading_tol_deg=HEADING_TOL_DEG,
        speed_tol=SPEED_TOL
    )
    print("\n[Q5 | 结果概览]")
    for k_, v in ans.items():
        if k_ != "rows":
            print(" ", k_, ":", v)

    print("\n[Q5 | 报表] 无人机×烟幕投放×导弹干扰明细（12列，含零遮蔽行）：")
    print_report_rows(ans["rows"])

    validate_q5(ans,
                n_trials=100,
                local_sigma_t=0.6,
                local_sigma_tau_rel=0.30,
                dt_list=(0.010,0.015,0.020,0.030),
                NANG_list=(24,48,96),
                NZ_list=(5,9,13),
                curve_sig_t=(0.0,0.2,0.4,0.6,0.8,1.0),
                curve_reps=30)

    neighborhood_validate_best(ans,
                               n_samples=100,
                               eps_t=0.30,
                               eps_tau=0.20,
                               ratio_per_shot_jitter=0.25,
                               dt=DT_STEP, N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
                               seed=20250)
