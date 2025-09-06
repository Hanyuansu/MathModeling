# -*- coding: utf-8 -*-
"""
CUMCM 2025 A 题 —— 单机对三弹联合最优（Q2 → Q2.5 扩展版，含“≥1s 最小投放间隔”硬约束）

- 变量：x = [theta, v, t1, tau1, t2, tau2, t3, tau3]
- 目标：最大化覆盖总时长（M1/M2/M3 的覆盖时长之和），不惩罚间隔
- 约束：首发后航向θ与速度v锁定不变；相邻两次投放间隔 **≥1s**（硬约束）
- 判定：L0（球-线段）极快；L1（圆柱采样）稳定；two_stage 先快后稳
- 输出：总覆盖、每枚导弹的覆盖区间、三枚弹的参数与起爆点
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

# =========================
# 一、题面常数与场景参数
# =========================
g = 9.81
VM = 300.0
V_SINK = 3.0
R_SMOKE = 10.0
T_EFFECT = 20.0

R_TAR, H_TAR = 7.0, 10.0
CYL_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)
P_TARGET   = np.array([0.0, 200.0, 5.0], dtype=float)

# 三枚来袭导弹初始坐标
MISSILES = [
    {"name": "M1", "M0": np.array([20000.0,    0.0, 2000.0], dtype=float)},
    {"name": "M2", "M0": np.array([19000.0,  600.0, 2100.0], dtype=float)},
    {"name": "M3", "M0": np.array([18000.0, -600.0, 1900.0], dtype=float)},
]

# 单架 UAV（默认 FY1；需要时可替换）
U0 = np.array([17800.0, 0.0, 1800.0], dtype=float)

# =========================
# 二、基础运动学/几何
# =========================
def missile_hit_time(m0: np.ndarray) -> float:
    return float(np.linalg.norm(m0) / VM)

def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v/n if n>0 else v

def missile_pos(m0: np.ndarray, t: float) -> np.ndarray:
    d = unit(-m0)
    return m0 + VM * d * t

def uav_pos(u0: np.ndarray, theta: float, v_u: float, t: float) -> np.ndarray:
    hx, hy = math.cos(theta), math.sin(theta)
    return np.array([u0[0] + v_u*hx*t, u0[1] + v_u*hy*t, u0[2]], dtype=float)

def burst_point(u0: np.ndarray, theta: float, v_u: float, t_drop: float, tau: float) -> np.ndarray:
    """起爆点 = 投放点 + 水平惯性 + 自由落体"""
    hx, hy = math.cos(theta), math.sin(theta)
    r_drop = uav_pos(u0, theta, v_u, t_drop)
    horiz  = np.array([v_u*hx*tau, v_u*hy*tau, 0.0], dtype=float)
    vert   = np.array([0.0, 0.0, -0.5*g*tau*tau], dtype=float)
    return r_drop + horiz + vert

def smoke_center_after_burst(s_burst: np.ndarray, t: float, t_burst: float) -> np.ndarray:
    dz = -V_SINK * max(0.0, t - t_burst)
    return s_burst + np.array([0.0, 0.0, dz], dtype=float)

def point_to_segment_dist(P: np.ndarray, Q: np.ndarray, X: np.ndarray) -> float:
    """点 X 到线段 PQ 的最小距离"""
    v = Q - P; vv = float(np.dot(v, v))
    if vv == 0.0: return float(np.linalg.norm(X - P))
    a = float(np.dot(X - P, v)/vv); a = 0.0 if a<0 else (1.0 if a>1.0 else a)
    Y = P + a*v
    return float(np.linalg.norm(X - Y))

def clip(x, lo, hi): return lo if x<lo else (hi if x>hi else x)

# =========================
# 三、L1 采样点与时刻判定
# =========================
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
    pts=[cyl_points_top_bottom(N_ang)]
    if include_side: pts.append(cyl_points_side(N_ang, N_z))
    return np.concatenate(pts, axis=0)

def covered_L0_at_time(m0, p_target, s_burst, t_burst, t) -> bool:
    """L0：以圆柱中心点代表视轴，球-线段相交"""
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    return (point_to_segment_dist(p_target, m_t, s_t) <= R_SMOKE)

def covered_L1_at_time_vectorized(m0: np.ndarray, s_burst: np.ndarray, t_burst: float, t: float, PTS: np.ndarray) -> bool:
    """L1：圆柱表面采样判定（向量化）"""
    m_t = missile_pos(m0, t); s_t = smoke_center_after_burst(s_burst, t, t_burst)
    v = m_t - PTS;  w = s_t - PTS
    vv = np.sum(v*v, axis=1)
    alpha = np.divide(np.sum(w*v, axis=1), vv, out=np.zeros_like(vv), where=vv>0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    Y = PTS + alpha[:, None]*v
    dist = np.linalg.norm(s_t - Y, axis=1)
    return bool(np.any(dist <= R_SMOKE))

# =========================
# 四、时间网格 & 区间工具
# =========================
def time_grid(T_hit: float, dt: float) -> np.ndarray:
    return np.arange(0.0, T_hit + 1e-12, dt)

def canonize_intervals(intervals, t0, t1, dt, eps=1e-9):
    """裁剪到[t0,t1]并 snap 到 dt 网格，去毛刺/合并相接段"""
    if not intervals: return []
    intervals = sorted(intervals, key=lambda ab: ab[0])
    def snap(x): return t0 + round((x - t0)/dt)*dt
    out=[]
    for a,b in intervals:
        a=max(t0,min(t1,a)); b=max(t0,min(t1,b))
        a=snap(a); b=snap(b)
        if b<=a+eps: continue
        if not out: out.append((a,b))
        else:
            pa,pb=out[-1]
            if a<=pb+eps: out[-1]=(pa,max(pb,b))
            else: out.append((a,b))
    return out

def intervals_from_mask(mask: np.ndarray, tgrid: np.ndarray) -> List[Tuple[float,float]]:
    """把布尔掩码转换为连续区间列表"""
    out=[]; in_seg=False; a=None
    for k,flag in enumerate(mask):
        if flag and not in_seg:
            in_seg=True; a=float(tgrid[k])
        if in_seg and (k==len(mask)-1 or (not mask[k+1])):
            b=float(tgrid[k]); out.append((a,b)); in_seg=False
    return out

# =========================
# 五、★ 最小投放间隔硬约束（做法 A）
# =========================
def enforce_min_gap(pairs: List[Tuple[float,float]],
                    min_gap: float = 1.0,
                    t_min: float = 0.0,
                    t_max: float = 60.0) -> List[Tuple[float,float]]:
    """
    输入：[(t1,tau1), (t2,tau2), (t3,tau3)]
    步骤：
      1) 先按 t 升序；
      2) 前向“推后”：t[i] = max(t[i], t[i-1] + min_gap)；
      3) 若 t[-1] > t_max：整体左移到刚好贴着 t_max；
      4) 若左移后 t[0] < t_min：把 t[0]=t_min，再按 min_gap 前向重算一次。
    注：对 0..60 窗口、min_gap=1s、3 枚投放，始终能排得下。
    """
    if not pairs: return []
    # 裁剪并排序
    pairs_sorted = sorted([(clip(t, t_min, t_max), clip(ta, 0.2, 12.0)) for (t, ta) in pairs],
                          key=lambda x: x[0])
    ts  = [t for (t, _) in pairs_sorted]
    taus= [ta for (_, ta) in pairs_sorted]

    # 前向推后
    ts_adj = [max(ts[0], t_min)]
    for i in range(1, len(ts)):
        ts_adj.append(max(ts[i], ts_adj[i-1] + min_gap))

    # 超上界则整体左移
    overflow = ts_adj[-1] - t_max
    if overflow > 1e-9:
        ts_adj = [t - overflow for t in ts_adj]
        # 若左移后越下界，则再从 t_min 重新推后
        if ts_adj[0] < t_min:
            ts_adj[0] = t_min
            for i in range(1, len(ts_adj)):
                ts_adj[i] = max(ts_adj[i], ts_adj[i-1] + min_gap)

    # 最终裁剪
    ts_adj = [clip(t, t_min, t_max) for t in ts_adj]
    return list(zip(ts_adj, taus))

# =========================
# 六、联合覆盖评价（多弹、多次起爆）
# =========================
def eval_multi_cover_L0(theta: float, v_u: float, drops: List[Tuple[float,float]],
                        dt: float = 0.02, min_drop_gap: float = 1.0) -> Tuple[float, Dict[str,Any]]:
    """
    L0：对三枚导弹，三次起爆的“并集覆盖”（每枚导弹各自并集，再相加）
    强制“相邻投放 ≥ min_drop_gap”
    """
    theta = theta % (2.0*math.pi)
    v_u   = clip(v_u,   70.0, 140.0)
    drops = [(clip(td,0.0,60.0), clip(ta,0.2,12.0)) for (td,ta) in drops]
    drops = enforce_min_gap(drops, min_gap=min_drop_gap, t_min=0.0, t_max=60.0)   # ★ 硬约束

    # 预计算每枚导弹的时间网格与空掩码
    tgrids = {m["name"]: time_grid(missile_hit_time(m["M0"]), dt) for m in MISSILES}
    union  = {m["name"]: np.zeros_like(tgrids[m["name"]], dtype=bool) for m in MISSILES}

    # 逐起爆条目累积并集
    s_bursts=[]
    for (td,ta) in drops:
        t_b = td + ta
        sb  = burst_point(U0, theta, v_u, td, ta)
        s_bursts.append((t_b, sb))
        if sb[2] <= 0.0:
            continue
        for m in MISSILES:
            name, m0 = m["name"], m["M0"]
            T_HIT = missile_hit_time(m0)
            if t_b >= T_HIT:
                continue
            tgrid = tgrids[name]
            idx = np.where((tgrid >= t_b) & (tgrid <= min(t_b+T_EFFECT, T_HIT)))[0]
            for k in idx:
                t = float(tgrid[k])
                if covered_L0_at_time(m0, P_TARGET, sb, t_b, t):
                    union[name][k] = True

    # 覆盖统计与区间
    per_missile=[]
    total=0.0
    for m in MISSILES:
        name = m["name"]; tgrid=tgrids[name]; mask=union[name]
        cov = float(mask.sum()*dt)
        its = canonize_intervals(intervals_from_mask(mask, tgrid), t0=float(tgrid[0]), t1=float(tgrid[-1]), dt=dt)
        per_missile.append({"missile": name, "cover_s": round(cov,3), "intervals": [(round(a,3),round(b,3)) for a,b in its]})
        total += cov

    detail = {"tgrids": tgrids, "union_masks": union, "s_bursts": s_bursts}
    return float(total), {"per_missile": per_missile, "detail": detail}

def eval_multi_cover_L1(theta: float, v_u: float, drops: List[Tuple[float,float]],
                        PTS: np.ndarray, dt: float = 0.02, min_drop_gap: float = 1.0) -> Tuple[float, Dict[str,Any]]:
    """L1：与 L0 相同，只把单时刻判定换成 L1；同样强制最小间隔"""
    theta = theta % (2.0*math.pi)
    v_u   = clip(v_u,   70.0, 140.0)
    drops = [(clip(td,0.0,60.0), clip(ta,0.2,12.0)) for (td,ta) in drops]
    drops = enforce_min_gap(drops, min_gap=min_drop_gap, t_min=0.0, t_max=60.0)   # ★ 硬约束

    tgrids = {m["name"]: time_grid(missile_hit_time(m["M0"]), dt) for m in MISSILES}
    union  = {m["name"]: np.zeros_like(tgrids[m["name"]], dtype=bool) for m in MISSILES}

    s_bursts=[]
    for (td,ta) in drops:
        t_b = td + ta
        sb  = burst_point(U0, theta, v_u, td, ta)
        s_bursts.append((t_b, sb))
        if sb[2] <= 0.0:
            continue
        for m in MISSILES:
            name, m0 = m["name"], m["M0"]
            T_HIT = missile_hit_time(m0)
            if t_b >= T_HIT: continue
            tgrid = tgrids[name]
            idx = np.where((tgrid >= t_b) & (tgrid <= min(t_b+T_EFFECT, T_HIT)))[0]
            for k in idx:
                t = float(tgrid[k])
                if covered_L1_at_time_vectorized(m0, sb, t_b, t, PTS):
                    union[name][k] = True

    per_missile=[]; total=0.0
    for m in MISSILES:
        name = m["name"]; tgrid=tgrids[name]; mask=union[name]
        cov = float(mask.sum()*dt)
        its = canonize_intervals(intervals_from_mask(mask, tgrid), t0=float(tgrid[0]), t1=float(tgrid[-1]), dt=dt)
        per_missile.append({"missile": name, "cover_s": round(cov,3), "intervals": [(round(a,3),round(b,3)) for a,b in its]})
        total += cov

    detail = {"tgrids": tgrids, "union_masks": union, "s_bursts": s_bursts}
    return float(total), {"per_missile": per_missile, "detail": detail}

# =========================
# 七、PSO（连续变量）
# =========================
class PSO:
    """
    粒子群优化（连续变量）
    变量：x = [theta, v, t1, tau1, t2, tau2, t3, tau3]
    目标：最大化覆盖时长（实现用 -cover 作损失）
    """
    def __init__(self, f_eval, bounds, swarm_size=96, iters=160,
                 inertia_w=0.72, c1=1.49, c2=1.49, seed=2025, init_hint: Optional[np.ndarray]=None):
        self.f_eval=f_eval; self.bounds=bounds
        self.swarm_size=swarm_size; self.iters=iters
        self.w=inertia_w; self.c1=c1; self.c2=c2
        self.rng=np.random.default_rng(seed)
        self.init_hint=init_hint

    def _init(self):
        D=len(self.bounds)
        X=np.zeros((self.swarm_size, D), dtype=float)
        V=np.zeros((self.swarm_size, D), dtype=float)
        for j,(lo,hi) in enumerate(self.bounds):
            X[:,j]=self.rng.uniform(lo, hi, size=self.swarm_size)
            span=hi-lo
            V[:,j]=self.rng.uniform(-0.1*span, 0.1*span, size=self.swarm_size)
        if self.init_hint is not None and len(self.init_hint)==D:
            X[0]=self.init_hint.copy(); V[0]=0.0
        pbest_X=X.copy(); pbest_val=np.full(self.swarm_size, np.inf)
        gbest_x=None; gbest_val=np.inf; gbest_info=None
        return X,V,pbest_X,pbest_val,gbest_x,gbest_val,gbest_info

    def _loss(self, x):
        # 角度周期化；其余裁剪在 f_eval 内
        x = x.copy()
        x[0] = x[0] % (2.0*math.pi)
        cover, info = self.f_eval(x)
        return -float(cover), float(cover), info

    def optimize(self):
        X,V,pbest_X,pbest_val,gbest_x,gbest_val,gbest_info = self._init()
        # 初评估
        for i in range(self.swarm_size):
            loss, cov, info = self._loss(X[i])
            pbest_X[i]=X[i].copy(); pbest_val[i]=loss
            if loss<gbest_val: gbest_val=loss; gbest_x=X[i].copy(); gbest_info=info
        # 迭代
        for _ in range(self.iters):
            w=self.w
            for i in range(self.swarm_size):
                r1=self.rng.random(len(self.bounds)); r2=self.rng.random(len(self.bounds))
                V[i]=w*V[i] + self.c1*r1*(pbest_X[i]-X[i]) + self.c2*r2*(gbest_x-X[i])
                X[i]=X[i]+V[i]
                # 均匀边界处理
                for j,(lo,hi) in enumerate(self.bounds):
                    if j==0: X[i,j]=X[i,j]%(2.0*math.pi)
                    else:
                        if X[i,j]<lo or X[i,j]>hi:
                            X[i,j]=clip(X[i,j], lo, hi); V[i,j]*=-0.5
                loss, cov, info = self._loss(X[i])
                if loss<pbest_val[i]: pbest_val[i]=loss; pbest_X[i]=X[i].copy()
                if loss<gbest_val: gbest_val=loss; gbest_x=X[i].copy(); gbest_info=info
        return gbest_x, -gbest_val, gbest_info

# =========================
# 八、求解接口（L0 / L1 / two_stage）
# =========================
def solve_one_uav_three_missiles(
    strategy: str = "two_stage",
    # L0/L1 步长与采样
    dt_L0: float = 0.02,
    dt_L1: float = 0.02,
    N_ANG: int = 48, N_Z: int = 9, INCLUDE_SIDE: bool = True,
    # PSO 规模
    swarm_size: int = 96, iters: int = 160, seed: int = 2025,
    stage2_swarm: int = 64, stage2_iters: int = 100,
    # 约束
    min_drop_gap: float = 1.0,
    # 可选初始化（theta, v, t1, tau1, t2, tau2, t3, tau3）
    init_hint: Optional[Tuple[float,float,float,float,float,float,float,float]] = None
) -> Dict[str,Any]:

    # 决策变量边界
    bounds = [
        (0.0, 2.0*math.pi),  # theta
        (70.0, 140.0),       # v
        (0.0, 60.0), (0.2, 12.0),  # t1, tau1
        (0.0, 60.0), (0.2, 12.0),  # t2, tau2
        (0.0, 60.0), (0.2, 12.0),  # t3, tau3
    ]

    # L1 采样
    PTS=None
    if strategy in ("L1","two_stage"):
        PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)

    # 构造 f_eval（把 8 维向量拆回 (theta, v, [(t_i, tau_i)]*3)）
    def unpack(x):
        th=float(x[0]); v=float(x[1])
        drops=[(float(x[2]),float(x[3])), (float(x[4]),float(x[5])), (float(x[6]),float(x[7]))]
        return th, v, drops

    def f_eval_vec_L0(x):
        th,v,drops = unpack(x)
        return eval_multi_cover_L0(th, clip(v,70.0,140.0),
                                   [(clip(td,0.0,60.0), clip(ta,0.2,12.0)) for (td,ta) in drops],
                                   dt=dt_L0, min_drop_gap=min_drop_gap)

    def f_eval_vec_L1(x):
        th,v,drops = unpack(x)
        return eval_multi_cover_L1(th, clip(v,70.0,140.0),
                                   [(clip(td,0.0,60.0), clip(ta,0.2,12.0)) for (td,ta) in drops],
                                   PTS=PTS, dt=dt_L1, min_drop_gap=min_drop_gap)

    # init_hint
    init_vec=None
    if init_hint is not None:
        th,v,t1,a1,t2,a2,t3,a3 = init_hint
        init_vec = np.array([
            float(th)%(2.0*math.pi),
            clip(float(v), 70.0, 140.0),
            clip(float(t1),0.0,60.0), clip(float(a1),0.2,12.0),
            clip(float(t2),0.0,60.0), clip(float(a2),0.2,12.0),
            clip(float(t3),0.0,60.0), clip(float(a3),0.2,12.0),
        ], dtype=float)

    # 优化
    if strategy=="L0":
        pso = PSO(lambda x: f_eval_vec_L0(x), bounds, swarm_size=swarm_size, iters=iters, seed=seed, init_hint=init_vec)
        best_x, best_cov, info = pso.optimize()
        eval_used="L0"
    elif strategy=="L1":
        pso = PSO(lambda x: f_eval_vec_L1(x), bounds, swarm_size=swarm_size, iters=iters, seed=seed, init_hint=init_vec)
        best_x, best_cov, info = pso.optimize()
        eval_used="L1"
    elif strategy=="two_stage":
        pso1 = PSO(lambda x: f_eval_vec_L0(x), bounds, swarm_size=swarm_size, iters=iters, seed=seed, init_hint=init_vec)
        x1, cov1, _ = pso1.optimize()
        pso2 = PSO(lambda x: f_eval_vec_L1(x), bounds, swarm_size=stage2_swarm, iters=stage2_iters, seed=seed+1, init_hint=x1)
        best_x, best_cov, info = pso2.optimize()
        eval_used="two_stage"
    else:
        raise ValueError("strategy 必须为 'L0'、'L1' 或 'two_stage'")

    # 组装输出
    th,v = float(best_x[0]), float(best_x[1])
    drops=[(float(best_x[2]),float(best_x[3])), (float(best_x[4]),float(best_x[5])), (float(best_x[6]),float(best_x[7]))]
    # 输出里也显示“硬约束后”的最终投放时刻（便于核验）
    drops_show = enforce_min_gap([(clip(td,0.0,60.0), clip(ta,0.2,12.0)) for (td,ta) in drops],
                                 min_gap=min_drop_gap, t_min=0.0, t_max=60.0)

    bursts=[]
    for (td,ta) in drops_show:
        t_b=td+ta; sb=burst_point(U0, th, v, td, ta)
        bursts.append({"t_drop": round(td,3), "tau": round(ta,3), "t_burst": round(t_b,3),
                       "s_burst": (round(float(sb[0]),3), round(float(sb[1]),3), round(float(sb[2]),3))})

    result = {
        "strategy": eval_used,
        "theta_deg": round((math.degrees(th)%360.0), 3),
        "v_u_mps": round(v, 3),
        "bursts": bursts,                        # 3 枚弹的参数（已满足 ≥1s）
        "total_cover_s": round(float(best_cov), 3),
        "per_missile": info["per_missile"],      # 每枚导弹的覆盖与区间（并集）
        "config": {
            "dt_L0": dt_L0, "dt_L1": dt_L1,
            "N_ANG": N_ANG, "N_Z": N_Z, "INCLUDE_SIDE": INCLUDE_SIDE,
            "swarm_size": swarm_size, "iters": iters,
            "stage2_swarm": stage2_swarm, "stage2_iters": stage2_iters,
            "min_drop_gap": min_drop_gap
        }
    }
    return result

# =========================
# 九、示例运行（只打印）
# =========================
if __name__ == "__main__":
    ans = solve_one_uav_three_missiles(
        strategy="two_stage",
        dt_L0=0.02, dt_L1=0.02,
        N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
        swarm_size=96, iters=160, stage2_swarm=64, stage2_iters=100,
        min_drop_gap=1.0
    )

    print("\n[单机对三弹 | 结果摘要]")
    print("  航向θ(°):", ans["theta_deg"])
    print("  速度v(m/s):", ans["v_u_mps"])
    print("  三枚烟幕弹（已满足 ≥1s）：(t_drop, tau, t_burst, s_burst)")
    for i,b in enumerate(ans["bursts"], start=1):
        print(f"    #{i}: t_drop={b['t_drop']}, tau={b['tau']}, t_burst={b['t_burst']}, s_burst={b['s_burst']}")
    print("  每枚导弹覆盖（并集）:")
    for item in ans["per_missile"]:
        print("    ", item)
    print("  覆盖总时长 (s):", ans["total_cover_s"])
