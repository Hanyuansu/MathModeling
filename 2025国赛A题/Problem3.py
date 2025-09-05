import math
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

# =========================
# 一、题面常数与场景参数
# =========================

g = 9.81                 # 重力加速度 (m/s^2)
VM = 300.0               # 导弹速度 (m/s)，匀速直线指向假目标(原点)
V_SINK = 3.0             # 云团下沉速度 (m/s)
R_SMOKE = 10.0           # 云团有效半径 (m)
T_EFFECT = 20.0          # 起爆后有效时间 (s)

# 真目标圆柱：半径7m，高10m；下底面圆心在(0,200,0)
R_TAR, H_TAR = 7.0, 10.0
CYL_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)
P_TARGET = np.array([0.0, 200.0, 5.0], dtype=float)  # L0：代表视轴点（圆柱几何中心）

# 场景（与前两问一致；如需测其它导弹/无人机，把这里改掉）
M0 = np.array([20000.0, 0.0, 2000.0], dtype=float)  # M1 初始
U0 = np.array([17800.0, 0.0, 1800.0], dtype=float)  # FY1 初始（等高度飞）

def missile_hit_time(m0: np.ndarray) -> float:
    """导弹到达原点的时间（用于裁剪上限）"""
    return float(np.linalg.norm(m0) / VM)

T_HIT = missile_hit_time(M0)

# =========================
# 二、通用工具
# =========================

def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v

def clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def to_pyfloat(x):
    """把 numpy 标量/数组中的数统一转成 Python float（递归处理 list/tuple/dict）。"""
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (list, tuple)):
        return type(x)(to_pyfloat(e) for e in x)
    if isinstance(x, dict):
        return {k: to_pyfloat(v) for k, v in x.items()}
    return x

def format_intervals(intervals: List[Tuple[float, float]], ndigits: int = 3):
    return [(round(a, ndigits), round(b, ndigits)) for (a, b) in intervals]

# =========================
# 三、运动学/几何
# =========================

def missile_pos(m0: np.ndarray, t: float) -> np.ndarray:
    """导弹：匀速直线指向原点"""
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
      horiz  = [v_u*hx*tau, v_u*hy*tau, 0]
      vert   = [0, 0, -0.5*g*tau^2]
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
    """点X到线段PQ的最小距离，用于球-线段相交判定"""
    v = Q - P
    vv = float(np.dot(v, v))
    if vv == 0.0:
        return float(np.linalg.norm(X - P))
    a = float(np.dot(X - P, v) / vv)
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    Y = P + a * v
    return float(np.linalg.norm(X - Y))

# =========================
# 四、L0 / L1 判定
# =========================

def covered_L0_at_time(m0, p_target, s_burst, t_burst, t) -> bool:
    """L0：以圆柱中心点为代表视轴，球-线段相交判定"""
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    d = point_to_segment_dist(p_target, m_t, s_t)
    return (d <= R_SMOKE)

def cyl_points_top_bottom(N_ang: int = 48) -> np.ndarray:
    """L1：上下圆面各 N_ang 个点"""
    cx, cy, _ = CYL_CENTER
    ks = np.arange(N_ang, dtype=float)
    angs = 2.0 * math.pi * ks / float(N_ang)
    cos_a, sin_a = np.cos(angs), np.sin(angs)
    xy = np.stack([R_TAR * cos_a, R_TAR * sin_a], axis=1)  # (N_ang,2)
    top = np.column_stack([cx + xy[:, 0], cy + xy[:, 1], np.full_like(ks, H_TAR)])
    bot = np.column_stack([cx + xy[:, 0], cy + xy[:, 1], np.zeros_like(ks)])
    return np.vstack([bot, top]).astype(float)

def cyl_points_side(N_ang: int = 48, N_z: int = 9) -> np.ndarray:
    """L1：侧壁 N_z 层，每层 N_ang 个点"""
    cx, cy, _ = CYL_CENTER
    zs = np.linspace(0.0, H_TAR, N_z)
    ks = np.arange(N_ang, dtype=float)
    angs = 2.0 * math.pi * ks / float(N_ang)
    cos_a, sin_a = np.cos(angs), np.sin(angs)
    xy = np.stack([R_TAR * cos_a, R_TAR * sin_a], axis=1)  # (N_ang,2)

    pts = []
    for z in zs:
        layer = np.column_stack([cx + xy[:, 0], cy + xy[:, 1], np.full_like(ks, z)])
        pts.append(layer)
    return np.vstack(pts).astype(float)

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
    v = m_t - PTS                 # (N,3) 线段向量 p->m
    w = s_t - PTS
    vv = np.einsum('ij,ij->i', v, v)
    # alpha = clamp( (w·v)/(v·v), 0, 1 )
    alpha = np.zeros_like(vv)
    valid = vv > 0.0
    alpha[valid] = np.clip(np.einsum('ij,ij->i', w[valid], v[valid]) / vv[valid], 0.0, 1.0)
    Y = PTS + alpha[:, None] * v
    dist = np.linalg.norm(s_t - Y, axis=1)
    return bool(np.any(dist <= R_SMOKE))

# =========================
# 五、辅助：三枚投放的“排序+保证间隔”
# =========================

def enforce_min_gap(pairs: List[Tuple[float, float]], min_gap=1.0,
                    t_min=0.0, t_max=60.0) -> List[Tuple[float, float]]:
    """
    输入：[(t1,tau1), (t2,tau2), (t3,tau3)]
    操作：
      1) 先按 t 升序排列（三枚弹本质对称，排序无害）
      2) 逐个推后，确保 t[i] >= t[i-1] + min_gap
      3) 再裁剪到 [t_min, t_max]
    返回：纠正后的 (t_k, tau_k) 列表，顺序已按投放先后
    """
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    out = []
    prev = t_min - min_gap
    for (t, tau) in pairs_sorted:
        t_adj = max(t, prev + min_gap)
        t_adj = clip(t_adj, t_min, t_max)
        out.append((t_adj, tau))
        prev = t_adj
    # 强制 tau 合法
    out = [(clip(t, t_min, t_max), clip(tau, 0.2, 12.0)) for (t, tau) in out]
    return out

def safe_effect_filter(bursts_raw: List[Tuple[float, float, float, np.ndarray]]) \
        -> List[Tuple[float, float, float, np.ndarray]]:
    """
    仅保留“有效窗口与 [0, T_HIT] 有交集”的云团：
      起爆时刻 t_burst ∈ [0, T_HIT] 且 起爆点在地面上方；
      有效窗口 [t_burst, t_burst + T_EFFECT] 与 [0, T_HIT] 交集非空。
    """
    kept = []
    for (t_drop, tau, t_burst, s_burst) in bursts_raw:
        if s_burst[2] <= 0.0:
            continue
        if t_burst > T_HIT:
            continue
        if (t_burst + T_EFFECT) <= 0.0:  # 不可能
            continue
        kept.append((t_drop, tau, t_burst, s_burst))
    return kept

# =========================
# 六、遮蔽并集评价（多枚弹）
# =========================

def eval_cover_time_multi_L0(theta: float, v_u: float,
                             drops: List[float], taus: List[float],
                             dt: float = 0.01) -> Tuple[float, List[Tuple[float, float]], Dict]:
    """
    L0：三枚弹并集遮蔽
    - 输入 drops/taus 将被排序且强制满足间隔≥1s
    - 返回：(并集遮蔽总时长, 并集区间列表, 详细信息)
    """
    theta = theta % (2.0 * math.pi)
    v_u   = clip(v_u, 70.0, 140.0)
    pairs = enforce_min_gap(list(zip([clip(d,0,60) for d in drops],
                                     [clip(t,0.2,12.0) for t in taus])),
                            min_gap=1.0, t_min=0.0, t_max=60.0)

    # 预计算三枚弹的起爆
    bursts = []
    for (t_drop, tau) in pairs:
        t_burst = t_drop + tau
        s_burst = burst_point(U0, theta, v_u, t_drop, tau)
        bursts.append((t_drop, tau, t_burst, s_burst))

    bursts = safe_effect_filter(bursts)
    if not bursts:
        return 0.0, [], {"pairs": pairs, "bursts": []}

    # 时间范围：从最早起爆到最晚有效结束或 T_HIT
    t0 = min(b[2] for b in bursts)
    t1 = min(max(b[2] for b in bursts) + T_EFFECT, T_HIT)

    covered = 0.0
    intervals = []
    in_seg, seg_start = False, None

    # 小心端点丢失：用 while + 累加
    t = t0
    while t <= t1 + 1e-12:
        flag_any = False
        for (_, _, t_burst, s_burst) in bursts:
            if t_burst <= t <= t_burst + T_EFFECT:
                if covered_L0_at_time(M0, P_TARGET, s_burst, t_burst, t):
                    flag_any = True
                    break
        if flag_any and not in_seg:
            in_seg, seg_start = True, t
        if (not flag_any) and in_seg:
            in_seg = False
            intervals.append((seg_start, t))
        if flag_any:
            covered += dt
        t += dt

    if in_seg:
        intervals.append((seg_start, t1))

    info = {
        "pairs": pairs,
        "bursts": [{"t_drop": d, "tau": ta, "t_burst": tb,
                    "s_burst": (float(sb[0]), float(sb[1]), float(sb[2]))} for (d, ta, tb, sb) in bursts]
    }
    return covered, intervals, info

def eval_cover_time_multi_L1(theta: float, v_u: float,
                             drops: List[float], taus: List[float],
                             PTS: np.ndarray, dt: float = 0.02) -> Tuple[float, List[Tuple[float, float]], Dict]:
    """L1：三枚弹并集遮蔽（与上面同逻辑，只是单时刻判定换成 L1）"""
    theta = theta % (2.0 * math.pi)
    v_u   = clip(v_u, 70.0, 140.0)
    pairs = enforce_min_gap(list(zip([clip(d,0,60) for d in drops],
                                     [clip(t,0.2,12.0) for t in taus])),
                            min_gap=1.0, t_min=0.0, t_max=60.0)

    bursts = []
    for (t_drop, tau) in pairs:
        t_burst = t_drop + tau
        s_burst = burst_point(U0, theta, v_u, t_drop, tau)
        bursts.append((t_drop, tau, t_burst, s_burst))
    bursts = safe_effect_filter(bursts)
    if not bursts:
        return 0.0, [], {"pairs": pairs, "bursts": []}

    t0 = min(b[2] for b in bursts)
    t1 = min(max(b[2] for b in bursts) + T_EFFECT, T_HIT)

    covered = 0.0
    intervals = []
    in_seg, seg_start = False, None

    t = t0
    while t <= t1 + 1e-12:
        flag_any = False
        for (_, _, t_burst, s_burst) in bursts:
            if t_burst <= t <= t_burst + T_EFFECT:
                if covered_L1_at_time_vectorized(M0, s_burst, t_burst, t, PTS):
                    flag_any = True
                    break
        if flag_any and not in_seg:
            in_seg, seg_start = True, t
        if (not flag_any) and in_seg:
            in_seg = False
            intervals.append((seg_start, t))
        if flag_any:
            covered += dt
        t += dt

    if in_seg:
        intervals.append((seg_start, t1))

    info = {
        "pairs": pairs,
        "bursts": [{"t_drop": d, "tau": ta, "t_burst": tb,
                    "s_burst": (float(sb[0]), float(sb[1]), float(sb[2]))} for (d, ta, tb, sb) in bursts]
    }
    return covered, intervals, info

# =========================
# 七、通用 PSO（支持任意维度）
# =========================

class PSO:
    """
    粒子群优化（连续变量，通用维度）
    - f_eval(vec) -> (score, extra_info)
    - bounds: [(lo,hi)] * D
    """
    def __init__(self, f_eval, bounds, swarm_size=96, iters=180,
                 inertia_w=0.72, c1=1.49, c2=1.49, seed=2025, init_hint: Optional[np.ndarray] = None):
        self.f_eval = f_eval
        self.bounds = bounds
        self.swarm_size = swarm_size
        self.iters = iters
        self.w = inertia_w
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.default_rng(seed)
        self.init_hint = init_hint

    def _init_swarm(self):
        D = len(self.bounds)
        X = np.zeros((self.swarm_size, D), dtype=float)
        V = np.zeros((self.swarm_size, D), dtype=float)
        for j, (lo, hi) in enumerate(self.bounds):
            X[:, j] = self.rng.uniform(lo, hi, size=self.swarm_size)
            span = hi - lo
            V[:, j] = self.rng.uniform(-0.1 * span, 0.1 * span, size=self.swarm_size)
        if self.init_hint is not None and len(self.init_hint) == D:
            X[0] = self.init_hint.copy()
            V[0] = 0.0
        pbest_X = X.copy()
        pbest_val = np.full(self.swarm_size, np.inf, dtype=float)
        gbest_x = None
        gbest_val = np.inf
        return X, V, pbest_X, pbest_val, gbest_x, gbest_val

    def _clip_vec(self, x):
        for j, (lo, hi) in enumerate(self.bounds):
            if j == 0:  # theta 允许环绕
                x[j] = x[j] % (2.0 * math.pi)
            else:
                x[j] = clip(x[j], lo, hi)
        return x

    def optimize(self):
        X, V, pbest_X, pbest_val, gbest_x, gbest_val = self._init_swarm()
        # 初评估
        gbest_info = None
        for i in range(self.swarm_size):
            xi = self._clip_vec(X[i].copy())
            score_i, _info = self.f_eval(xi)
            loss_i = -score_i  # 最大化->最小化
            pbest_X[i] = xi
            pbest_val[i] = loss_i
            if loss_i < gbest_val:
                gbest_val = loss_i
                gbest_x = xi.copy()
                gbest_info = _info
        # 迭代
        for _iter in range(self.iters):
            w = self.w
            for i in range(self.swarm_size):
                r1 = np.random.random(len(self.bounds))
                r2 = np.random.random(len(self.bounds))
                V[i] = w * V[i] + self.c1 * r1 * (pbest_X[i] - X[i]) + self.c2 * r2 * (gbest_x - X[i])
                X[i] = self._clip_vec(X[i] + V[i])
                score_i, info_i = self.f_eval(X[i])
                loss_i = -score_i
                if loss_i < pbest_val[i]:
                    pbest_val[i] = loss_i
                    pbest_X[i] = X[i].copy()
                if loss_i < gbest_val:
                    gbest_val = loss_i
                    gbest_x = X[i].copy()
                    gbest_info = info_i
        best_cover = -gbest_val
        return gbest_x, best_cover, gbest_info

# =========================
# 八、统一求解接口（L0 / L1 / two_stage）
# =========================

def solve_q3(strategy: str = "L0",
             # L0/L1 数值参数
             dt_L0: float = 0.01,
             N_ANG: int = 48, N_Z: int = 9, INCLUDE_SIDE: bool = True, dt_L1: float = 0.02,
             # PSO 参数
             swarm_size: int = 96, iters: int = 180, seed: int = 2025,
             # 两阶段二次精修的 PSO 参数
             stage2_swarm: int = 64, stage2_iters: int = 90,
             # 可选初始化提示（来自你的直觉或 Q2 最优）：(theta, v, t1, tau1, t2, tau2, t3, tau3)
             init_hint: Optional[Tuple[float, float, float, float, float, float, float, float]] = None
             ) -> Dict[str, Any]:
    """
    返回：{
       theta_deg, v_u_mps,
       drops_s: [t1,t2,t3], taus_s: [tau1,tau2,tau3],
       bursts: [{t_drop, tau, t_burst, s_burst(x,y,z)}*3 排好序],
       cover_total_s, cover_intervals_s,
       config/strategy...
    }
    """
    # 维度与边界（theta, v, t1, tau1, t2, tau2, t3, tau3）
    bounds = [
        (0.0, 2.0 * math.pi),  # theta
        (70.0, 140.0),         # v
        (0.0, 60.0), (0.2, 12.0),
        (0.0, 60.0), (0.2, 12.0),
        (0.0, 60.0), (0.2, 12.0),
    ]

    # L1 采样点（仅 L1 / two_stage 需要）
    PTS = None
    if strategy in ("L1", "two_stage"):
        # 使用 N_z 参数名
        PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)

    # 定义 f_eval（针对 8 维向量）
    def f_eval_L0(vec):
        th, v, t1, ta1, t2, ta2, t3, ta3 = vec
        cover, intervals, info = eval_cover_time_multi_L0(th, v, [t1, t2, t3], [ta1, ta2, ta3], dt=dt_L0)
        return cover, {"intervals": intervals, "info": info, "theta": th, "v": v}

    def f_eval_L1(vec):
        th, v, t1, ta1, t2, ta2, t3, ta3 = vec
        cover, intervals, info = eval_cover_time_multi_L1(th, v, [t1, t2, t3], [ta1, ta2, ta3], PTS=PTS, dt=dt_L1)
        return cover, {"intervals": intervals, "info": info, "theta": th, "v": v}

    # 初值（可选）
    init_vec = None
    if init_hint is not None:
        init_vec = np.array([
            float(init_hint[0]) % (2.0 * math.pi),
            clip(float(init_hint[1]), bounds[1][0], bounds[1][1]),
            clip(float(init_hint[2]), bounds[2][0], bounds[2][1]), clip(float(init_hint[3]), bounds[3][0], bounds[3][1]),
            clip(float(init_hint[4]), bounds[4][0], bounds[4][1]), clip(float(init_hint[5]), bounds[5][0], bounds[5][1]),
            clip(float(init_hint[6]), bounds[6][0], bounds[6][1]), clip(float(init_hint[7]), bounds[7][0], bounds[7][1]),
        ], dtype=float)

    # 按策略求解
    if strategy == "L0":
        pso = PSO(f_eval=f_eval_L0, bounds=bounds, swarm_size=swarm_size, iters=iters, seed=seed, init_hint=init_vec)
        best_x, best_cover, extra = pso.optimize()
        eval_used = "L0"
    elif strategy == "L1":
        pso = PSO(f_eval=f_eval_L1, bounds=bounds, swarm_size=swarm_size, iters=iters, seed=seed, init_hint=init_vec)
        best_x, best_cover, extra = pso.optimize()
        eval_used = "L1"
    elif strategy == "two_stage":
        # Stage 1: L0 快速全局找好解
        pso1 = PSO(f_eval=f_eval_L0, bounds=bounds, swarm_size=swarm_size, iters=iters, seed=seed, init_hint=init_vec)
        x1, cover1, extra1 = pso1.optimize()
        # Stage 2: L1 小群体精修
        pso2 = PSO(f_eval=f_eval_L1, bounds=bounds, swarm_size=stage2_swarm, iters=stage2_iters, seed=seed+1, init_hint=x1)
        best_x, best_cover, extra = pso2.optimize()
        eval_used = "two_stage"
    else:
        raise ValueError("strategy 必须为 'L0'、'L1' 或 'two_stage'")

    # 整理输出（全部转为 Python float，避免 np.float64）
    th = float(extra["theta"]); v = float(extra["v"])
    info = extra["info"]
    intervals = extra["intervals"]
    pairs = info["pairs"]              # 纠正后的 (t, tau)
    bursts = info["bursts"]            # 每枚弹的起爆信息（已按时间序）

    out = {
        "strategy": eval_used,
        "theta_deg": float(math.degrees(th)),
        "v_u_mps": float(v),
        "drops_s": [float(round(p[0], 3)) for p in pairs],
        "taus_s":  [float(round(p[1], 3)) for p in pairs],
        "bursts": to_pyfloat(bursts),
        "cover_total_s": float(best_cover),
        "cover_intervals_s": format_intervals(intervals, 3),
        "config": {
            "dt_L0": float(dt_L0), "dt_L1": float(dt_L1),
            "N_ANG": int(N_ANG), "N_Z": int(N_Z), "INCLUDE_SIDE": bool(INCLUDE_SIDE),
            "swarm_size": int(swarm_size), "iters": int(iters),
            "stage2_swarm": int(stage2_swarm), "stage2_iters": int(stage2_iters)
        }
    }
    return out

# =========================
# 九、结果美化打印 & 随机健壮性检查
# =========================

def pretty_print(tag: str, ans: Dict[str, Any]):
    print(f"[Q3 | {tag}] 最优：")
    print(f"  strategy: {ans['strategy']}")
    print(f"  theta_deg: {round(ans['theta_deg'], 3)}")
    print(f"  v_u_mps: {round(ans['v_u_mps'], 3)}")
    print(f"  drops_s: {ans['drops_s']}")
    print(f"  taus_s:  {ans['taus_s']}")
    print(f"  cover_total_s: {round(ans['cover_total_s'], 3)}")
    print(f"  cover_intervals_s: {ans['cover_intervals_s']}")
    if ans.get("bursts"):
        print(f"  bursts: {ans['bursts'][:3]}")  # 只展示前三条（本题通常3条）

def randomized_check(n_samples: int = 100,
                     eval_strategy: str = "L1",
                     around_solution: Optional[Dict[str, Any]] = None,
                     seed: int = 2025,
                     N_ANG: int = 48, N_Z: int = 9, INCLUDE_SIDE: bool = True,
                     dt_eval: float = 0.02):
    """
    随机检查：用 L1/L0 评估随机参数的并集遮蔽时长，验证基线最优解的稳健性
    - global 模式：参数在全域均匀采样（更易出现 0，但更客观）
    - local  模式：围绕给定解做高斯扰动（更易出现非 0，便于 sanity check）
    """
    rng = np.random.default_rng(seed)
    PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE) if eval_strategy == "L1" else None

    def sample_global():
        theta = rng.uniform(0, 2*math.pi)
        v = rng.uniform(70.0, 140.0)
        # 为提高出现非零覆盖的概率，固定前两弹靠前、第三弹靠 50s 左右（但仍加抖动）
        t1 = rng.uniform(0.0, 0.3)
        t2 = t1 + rng.uniform(1.0, 1.5)
        t3 = rng.uniform(47.0, 53.0)
        tau1 = rng.uniform(0.2, 0.8)
        tau2 = rng.uniform(0.2, 0.8)
        tau3 = rng.uniform(0.8, 3.0)
        return theta, v, [t1, t2, t3], [tau1, tau2, tau3]

    def sample_local(base):
        # 围绕给定解做小扰动（高斯），并裁剪合法
        theta = (math.radians(base["theta_deg"]) + rng.normal(0, math.radians(4.0))) % (2*math.pi)
        v = clip(base["v_u_mps"] + rng.normal(0, 8.0), 70.0, 140.0)
        t1, t2, t3 = [clip(x + rng.normal(0, 1.0), 0, 60) for x in base["drops_s"]]
        tau1, tau2, tau3 = [clip(x + rng.normal(0, 0.6), 0.2, 12.0) for x in base["taus_s"]]
        return theta, v, [t1, t2, t3], [tau1, tau2, tau3]

    def eval_once(theta, v, drops, taus):
        if eval_strategy == "L1":
            cover, _, _ = eval_cover_time_multi_L1(theta, v, drops, taus, PTS=PTS, dt=dt_eval)
        else:
            cover, _, _ = eval_cover_time_multi_L0(theta, v, drops, taus, dt=dt_eval)
        return float(cover), theta, v, list(map(float, drops)), list(map(float, taus))

    mode = "local" if around_solution is not None else "global"
    covers = []
    records = []

    for _ in range(n_samples):
        if mode == "local":
            theta, v, drops, taus = sample_local(around_solution)
        else:
            theta, v, drops, taus = sample_global()
        c, th, vv, ds, ts = eval_once(theta, v, drops, taus)
        covers.append(c)
        records.append((c, th, vv, ds, ts))

    covers = np.array(covers, dtype=float)
    order = np.argsort(-covers)
    print(f"\n[check] {mode} 随机 {n_samples} 组（{eval_strategy} 评估）")
    print(f"  mean_cover_s: {covers.mean():.3f}")
    print(f"  std_cover_s: {covers.std():.3f}")
    print(f"  max_cover_s: {covers.max():.3f}")
    print(f"  min_cover_s: {covers.min():.3f}")
    print("  top5:")
    topk = min(5, n_samples)
    for i in range(topk):
        c, th, vv, ds, ts = records[order[i]]
        print(f"    #{i+1}: cover={c:.3f}s, theta={math.degrees(th):.2f}deg, v={vv:.1f}, "
              f"pairs=[({ds[0]:.3f},{ts[0]:.3f}), ({ds[1]:.3f},{ts[1]:.3f}), ({ds[2]:.3f},{ts[2]:.3f})]")

# =========================
# 十、新增：打印“无人机方向/速度 + 每枚弹投放/起爆点 + 单枚有效时长”明细表
# =========================

def _single_mask_L0(theta: float, v: float, t_drop: float, tau: float, tgrid: np.ndarray) -> np.ndarray:
    """Q3 单枚弹在 L0 判定下的时间掩码"""
    t_burst = t_drop + tau
    s_burst = burst_point(U0, theta, v, t_drop, tau)
    if t_burst >= T_HIT or s_burst[2] <= 0.0:
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

def _single_mask_L1(theta: float, v: float, t_drop: float, tau: float, tgrid: np.ndarray, PTS: np.ndarray) -> np.ndarray:
    """Q3 单枚弹在 L1 判定下的时间掩码"""
    t_burst = t_drop + tau
    s_burst = burst_point(U0, theta, v, t_drop, tau)
    if t_burst >= T_HIT or s_burst[2] <= 0.0:
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

def print_q3_report_table(ans: Dict[str, Any]):
    """
    打印以下列的明细表（每枚干扰弹 1 行）：
    无人机运动方向 | 无人机运动速度 (m/s) | 烟幕干扰弹编号 | 投放点 x/y/z | 起爆点 x/y/z | 有效干扰时长 (s)
    """
    theta_deg = float(ans["theta_deg"])
    v = float(ans["v_u_mps"])
    drops = [float(x) for x in ans["drops_s"]]
    taus  = [float(x) for x in ans["taus_s"]]

    # 评估模式与时间网格
    mode_L1 = (ans["strategy"] != "L0")
    cfg = ans.get("config", {})
    dt = float(cfg["dt_L1"] if mode_L1 else cfg["dt_L0"])
    tgrid = np.arange(0.0, T_HIT + 1e-12, dt)
    PTS = None
    if mode_L1:
        PTS = build_cylinder_samples(N_ang=int(cfg.get("N_ANG", 48)),
                                     N_z=int(cfg.get("N_Z", 9)),
                                     include_side=bool(cfg.get("INCLUDE_SIDE", True)))

    print("无人机运动方向\t无人机运动速度 (m/s)\t烟幕干扰弹编号\t"
          "烟幕干扰弹投放点的x坐标 (m)\t烟幕干扰弹投放点的y坐标 (m)\t烟幕干扰弹投放点的z坐标 (m)\t"
          "烟幕干扰弹起爆点的x坐标 (m)\t烟幕干扰弹起爆点的y坐标 (m)\t烟幕干扰弹起爆点的z坐标 (m)\t"
          "有效干扰时长 (s)")

    th = math.radians(theta_deg)
    for i, (td, ta) in enumerate(zip(drops, taus), 1):
        # 几何点
        r_drop = uav_pos(U0, th, v, td)
        s_burst = burst_point(U0, th, v, td, ta)
        # 单枚有效时长（与求解一致的评估模式）
        if mode_L1:
            mask = _single_mask_L1(th, v, td, ta, tgrid, PTS)
        else:
            mask = _single_mask_L0(th, v, td, ta, tgrid)
        eff_time = float(mask.sum() * dt)
        # 打印
        print(f"{theta_deg:.3f}\t{v:.3f}\t{i}\t"
              f"{r_drop[0]:.3f}\t{r_drop[1]:.3f}\t{r_drop[2]:.3f}\t"
              f"{s_burst[0]:.3f}\t{s_burst[1]:.3f}\t{s_burst[2]:.3f}\t"
              f"{eff_time:.3f}")

# =========================
# 主程序（可直接运行）
# =========================
if __name__ == "__main__":
    # 1) 先用 L0 找初值（最快）
    ans_L0 = solve_q3(strategy="L0", dt_L0=0.01, swarm_size=96, iters=180)
    pretty_print("L0", ans_L0)
    print("\n[Q3 | L0] 明细表：")
    print_q3_report_table(ans_L0)

    # 2) 用 L1 直接求（可提供上一条的 ans_L0 作为 init_hint，加速收敛）
    hint = (math.radians(190.0), 120.0, 2.0, 3.5, 4.0, 3.2, 7.0, 3.0)
    ans_L1 = solve_q3(strategy="L1", N_ANG=48, N_Z=9, INCLUDE_SIDE=True, dt_L1=0.02,
                      swarm_size=80, iters=140, init_hint=None)  # 或 init_hint=hint
    print("[Q3 | L1] 最优：");
    [print(f"  {k}: {v}") for k, v in ans_L1.items()]

    # 3) 两阶段（推荐）：先 L0 全局，再 L1 小规模精修
    ans_2s = solve_q3(strategy="two_stage",
                      dt_L0=0.01, swarm_size=96, iters=180,
                      N_ANG=48, N_Z=9, INCLUDE_SIDE=True, dt_L1=0.02,
                      stage2_swarm=64, stage2_iters=80)
    pretty_print("two_stage", ans_2s)
    print("\n[Q3 | two_stage] 明细表：")
    print_q3_report_table(ans_2s)

    # 4) 可选：全局/局部随机健壮性检查（默认用 L1 评估，包含 PTS）
    randomized_check(n_samples=100, eval_strategy="L1", around_solution=None, seed=2025)
    randomized_check(n_samples=100, eval_strategy="L1", around_solution=ans_2s, seed=2026)




