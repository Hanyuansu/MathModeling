# -*- coding: utf-8 -*-
"""
CUMCM 2025 A题 —— 第4问：三架无人机（FY1、FY2、FY3）各投一枚的联合最优（融合版 · 完全体）

功能特性：
  - 判定策略：
      'L0'         —— 以目标圆柱“几何中心点”为代表视轴，速度快
      'L1'         —— 圆柱表面采样（上下圆面 + 侧壁），判定更稳健
      'two_stage'  —— 先 L0 全局搜索，再 L1 小规模精修（推荐）
  - 优化器：
      'PSO'（粒子群） | 'CEM'（交叉熵） | 'DE'（差分进化）
  - 几何引导初始化 make_q4_geometry_seed：让三枚云团对齐早/中/晚三段视线走廊
  - 单枚贡献诊断 per_cloud_coverage_report：定位哪一枚“贡献偏弱”

输出结构（solve_q4）：
{
  'strategy',                       # 实际采用的策略
  'theta_deg': [th1, th2, th3],     # 三机航向角（度）
  'v_u_mps':   [v1, v2, v3],        # 三机水平速度
  'bursts': [
      {uav,t_drop,tau,t_burst,s_burst=(x,y,z)},  # 按起爆时刻排序
      ...
  ],
  'cover_total_s',                  # 并集遮蔽总时长
  'cover_intervals_s': [(a,b), ...],# 并集遮蔽区间
  'config': {...}                   # 关键配置与超参
}

作者：ChatGPT（中文注释）
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

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
P_TARGET = np.array([0.0, 200.0, 5.0], dtype=float)  # L0代表点：圆柱几何中心

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
    """导弹命中原点的时刻：T_hit = ||m0|| / VM，用于截断积分上限"""
    return float(np.linalg.norm(m0) / VM)

T_HIT = missile_hit_time(M0)

# =========================
# 二、运动学/几何工具
# =========================

def unit(v: np.ndarray) -> np.ndarray:
    """单位向量（避免除零）"""
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
    """点X到线段PQ的最小距离（欧氏）"""
    v = Q - P
    vv = float(np.dot(v, v))
    if vv == 0.0:
        return float(np.linalg.norm(X - P))
    a = float(np.dot(X - P, v) / vv)
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    Y = P + a * v
    return float(np.linalg.norm(X - Y))

def clip(x, lo, hi):
    """数值裁剪到[lo, hi]"""
    return lo if x < lo else (hi if x > hi else x)

# =========================
# 三、L0/L1 单时刻判定
# =========================

def covered_L0_at_time(m0, p_target, s_burst, t_burst, t) -> bool:
    """L0：以圆柱中心点为代表轴，判定球-线段是否相交"""
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    return (point_to_segment_dist(p_target, m_t, s_t) <= R_SMOKE)

def cyl_points_top_bottom(N_ang: int = 48) -> np.ndarray:
    """L1：上下圆面各 N_ang 个点"""
    cx, cy, _ = CYL_CENTER
    pts = []
    for z in (0.0, H_TAR):
        for k in range(N_ang):
            ang = 2.0 * math.pi * k / N_ang
            pts.append((cx + R_TAR * math.cos(ang), cy + R_TAR * math.sin(ang), z))
    return np.array(pts, dtype=float)

def cyl_points_side(N_ang: int = 48, N_z: int = 9) -> np.ndarray:
    """L1：侧壁 N_z 层，每层 N_ang 个点"""
    cx, cy, _ = CYL_CENTER
    zs = np.linspace(0.0, H_TAR, N_z)
    pts = []
    for z in zs:
        for k in range(N_ang):
            ang = 2.0 * math.pi * k / N_ang
            pts.append((cx + R_TAR * math.cos(ang), cy + R_TAR * math.sin(ang), z))
    return np.array(pts, dtype=float)

def build_cylinder_samples(N_ang=48, N_z=9, include_side=True) -> np.ndarray:
    """构建 L1 采样点集合（上下圆面 + 侧壁可选）"""
    pts = [cyl_points_top_bottom(N_ang)]
    if include_side:
        pts.append(cyl_points_side(N_ang, N_z))
    return np.concatenate(pts, axis=0)

def covered_L1_at_time_vectorized(m0, s_burst, t_burst, t, PTS) -> bool:
    """L1：向量化判定“是否存在采样点被遮蔽”"""
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    v = m_t - PTS
    w = s_t - PTS
    vv = np.sum(v * v, axis=1)
    alpha = np.divide(np.sum(w * v, axis=1), vv, out=np.zeros_like(vv), where=vv > 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    Y = PTS + alpha[:, None] * v
    dist = np.linalg.norm(s_t - Y, axis=1)
    return bool(np.any(dist <= R_SMOKE))

# =========================
# 四、三架机并集遮蔽评价（核心）
# =========================

def _valid_burst(u0, theta, v, t_drop, tau):
    """
    生成单枚弹的起爆信息；若起爆点在地面/起爆过晚则返回 None
    返回：(t_drop, tau, t_burst, s_burst[3])
    """
    theta = theta % (2.0 * math.pi)
    v = clip(v, 70.0, 140.0)
    t_drop = clip(t_drop, 0.0, 60.0)
    tau = clip(tau, 0.2, 12.0)
    t_burst = t_drop + tau
    s_burst = burst_point(u0, theta, v, t_drop, tau)
    if s_burst[2] <= 0.0 or t_burst >= T_HIT:
        return None
    return (t_drop, tau, t_burst, s_burst)

def eval_cover_time_q4_L0(thetas: List[float], vs: List[float],
                          drops: List[float], taus: List[float],
                          dt: float = 0.01) -> Tuple[float, List[Tuple[float, float]], Dict]:
    """
    L0：三架机各一枚的并集遮蔽评价
    返回：(并集遮蔽总时长, 并集区间列表, 细节info)
    """
    thetas = [th % (2.0 * math.pi) for th in thetas]
    vs     = [clip(v, 70.0, 140.0) for v in vs]
    drops  = [clip(t, 0.0, 60.0) for t in drops]
    taus   = [clip(t, 0.2, 12.0) for t in taus]

    bursts = []
    for i in range(3):
        b = _valid_burst(UAVS[i]["U0"], thetas[i], vs[i], drops[i], taus[i])
        if b is not None:
            bursts.append({"uav": UAVS[i]["name"], "data": b})

    if not bursts:
        return 0.0, [], {"bursts": []}

    t0 = min(b["data"][2] for b in bursts)
    t1 = min(max(b["data"][2] for b in bursts) + T_EFFECT, T_HIT)

    covered = 0.0
    intervals, in_seg, seg_start = [], False, None

    t = t0
    while t <= t1 + 1e-12:
        flag = False
        for b in bursts:
            (t_drop, tau, t_burst, s_burst) = b["data"]
            if t_burst <= t <= t_burst + T_EFFECT:
                if covered_L0_at_time(M0, P_TARGET, s_burst, t_burst, t):
                    flag = True
                    break
        if flag and not in_seg:
            in_seg, seg_start = True, t
        if (not flag) and in_seg:
            in_seg = False
            intervals.append((seg_start, t))
        if flag:
            covered += dt
        t += dt

    if in_seg:
        intervals.append((seg_start, t1))

    info = {
        "bursts": [
            {"uav": b["uav"], "t_drop": float(b["data"][0]), "tau": float(b["data"][1]),
             "t_burst": float(b["data"][2]),
             "s_burst": (float(b["data"][3][0]), float(b["data"][3][1]), float(b["data"][3][2]))}
            for b in sorted(bursts, key=lambda x: x["data"][2])
        ]
    }
    return covered, intervals, info

def eval_cover_time_q4_L1(thetas: List[float], vs: List[float],
                          drops: List[float], taus: List[float],
                          PTS: np.ndarray, dt: float = 0.02) -> Tuple[float, List[Tuple[float, float]], Dict]:
    """L1：三架机各一枚的并集遮蔽（与上面同逻辑，单时刻判定换为 L1）"""
    thetas = [th % (2.0 * math.pi) for th in thetas]
    vs     = [clip(v, 70.0, 140.0) for v in vs]
    drops  = [clip(t, 0.0, 60.0) for t in drops]
    taus   = [clip(t, 0.2, 12.0) for t in taus]

    bursts = []
    for i in range(3):
        b = _valid_burst(UAVS[i]["U0"], thetas[i], vs[i], drops[i], taus[i])
        if b is not None:
            bursts.append({"uav": UAVS[i]["name"], "data": b})
    if not bursts:
        return 0.0, [], {"bursts": []}

    t0 = min(b["data"][2] for b in bursts)
    t1 = min(max(b["data"][2] for b in bursts) + T_EFFECT, T_HIT)

    covered = 0.0
    intervals, in_seg, seg_start = [], False, None

    t = t0
    while t <= t1 + 1e-12:
        flag = False
        for b in bursts:
            (t_drop, tau, t_burst, s_burst) = b["data"]
            if t_burst <= t <= t_burst + T_EFFECT:
                if covered_L1_at_time_vectorized(M0, s_burst, t_burst, t, PTS):
                    flag = True
                    break
        if flag and not in_seg:
            in_seg, seg_start = True, t
        if (not flag) and in_seg:
            in_seg = False
            intervals.append((seg_start, t))
        if flag:
            covered += dt
        t += dt

    if in_seg:
        intervals.append((seg_start, t1))

    info = {
        "bursts": [
            {"uav": b["uav"], "t_drop": float(b["data"][0]), "tau": float(b["data"][1]),
             "t_burst": float(b["data"][2]),
             "s_burst": (float(b["data"][3][0]), float(b["data"][3][1]), float(b["data"][3][2]))}
            for b in sorted(bursts, key=lambda x: x["data"][2])
        ]
    }
    return covered, intervals, info

# =========================
# 五、优化器：PSO / CEM / DE
# =========================

class PSO:
    """
    粒子群优化（连续变量，通用维度）
    - f_eval(vec) -> (score, extra_info)
    - bounds: [(lo,hi)] * D
    """
    def __init__(self, f_eval, bounds, swarm_size=128, iters=220,
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
            X[0] = self._clip_vec(self.init_hint.copy())
            V[0] = 0.0
        pbest_X = X.copy()
        pbest_val = np.full(self.swarm_size, np.inf, dtype=float)
        gbest_x = None
        gbest_val = np.inf
        return X, V, pbest_X, pbest_val, gbest_x, gbest_val

    def _clip_vec(self, x):
        for j, (lo, hi) in enumerate(self.bounds):
            if j % 4 == 0:  # theta 环绕
                x[j] = x[j] % (2.0 * math.pi)
            else:
                x[j] = clip(x[j], lo, hi)
        return x

    def optimize(self):
        X, V, pbest_X, pbest_val, gbest_x, gbest_val = self._init_swarm()

        # 初评估
        for i in range(self.swarm_size):
            xi = self._clip_vec(X[i].copy())
            score_i, _ = self.f_eval(xi)
            loss_i = -score_i
            pbest_X[i] = xi
            pbest_val[i] = loss_i
            if loss_i < gbest_val:
                gbest_val = loss_i
                gbest_x = xi.copy()
                gbest_info = _

        # 迭代
        for _iter in range(self.iters):
            for i in range(self.swarm_size):
                r1 = self.rng.random(len(self.bounds))
                r2 = self.rng.random(len(self.bounds))
                V[i] = self.w * V[i] + self.c1 * r1 * (pbest_X[i] - X[i]) + self.c2 * r2 * (gbest_x - X[i])
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


class CEM:
    """
    交叉熵方法（Cross-Entropy Method）
    - 适合非光滑/平台型目标（如遮蔽并集时长）
    """
    def __init__(self, f_eval, bounds, pop_size=240, iters=120,
                 elite_frac=0.15, use_full_cov=False,
                 alpha_mean=0.7, alpha_cov=0.5,
                 sigma0_frac=0.30, sigma_floor=1e-3,
                 seed=2025, init_hint: Optional[np.ndarray] = None):
        self.f_eval = f_eval
        self.bounds = bounds
        self.pop_size = pop_size
        self.iters = iters
        self.elite_frac = elite_frac
        self.use_full_cov = use_full_cov
        self.alpha_m = alpha_mean
        self.alpha_c = alpha_cov
        self.sigma0_frac = sigma0_frac
        self.sigma_floor = sigma_floor
        self.rng = np.random.default_rng(seed)
        self.init_hint = init_hint

    def _clip_vec(self, x):
        for j, (lo, hi) in enumerate(self.bounds):
            if j % 4 == 0:
                x[j] = x[j] % (2.0 * math.pi)
            else:
                x[j] = lo if x[j] < lo else (hi if x[j] > hi else x[j])
        return x

    def _init_dist(self):
        D = len(self.bounds)
        if self.init_hint is not None and len(self.init_hint) == D:
            m = self._clip_vec(self.init_hint.astype(float).copy())
        else:
            mids = [(lo + hi) * 0.5 for (lo, hi) in self.bounds]
            m = np.array(mids, dtype=float)
        spans = np.array([hi - lo for (lo, hi) in self.bounds], dtype=float)
        sig = np.maximum(self.sigma0_frac * spans, self.sigma_floor)
        C = np.diag(sig**2) if self.use_full_cov else sig**2
        return m, C

    def _sample(self, m, C, n):
        D = len(self.bounds)
        if self.use_full_cov:
            Z = self.rng.multivariate_normal(mean=np.zeros(D), cov=C, size=n)
            X = m[None, :] + Z
        else:
            Z = self.rng.normal(loc=0.0, scale=np.sqrt(C), size=(n, D))
            X = m[None, :] + Z
        for i in range(n):
            X[i] = self._clip_vec(X[i])
        return X

    def optimize(self):
        m, C = self._init_dist()
        elite_k = max(2, int(self.pop_size * self.elite_frac))
        best_x, best_score, best_extra = None, -np.inf, None

        for _ in range(self.iters):
            X = self._sample(m, C, self.pop_size)
            scores = np.empty(self.pop_size, dtype=float)
            extras = [None] * self.pop_size
            for i in range(self.pop_size):
                s, extra = self.f_eval(X[i])
                scores[i] = s
                extras[i] = extra
                if s > best_score:
                    best_score, best_x, best_extra = s, X[i].copy(), extra

            idx = np.argsort(scores)[::-1]
            elites = X[idx[:elite_k]]
            m_e = elites.mean(axis=0)
            if self.use_full_cov:
                centered = elites - m_e
                C_e = (centered.T @ centered) / max(1, elite_k - 1)
                C = (1 - self.alpha_c) * C + self.alpha_c * C_e
                for d in range(len(self.bounds)):
                    C[d, d] = max(C[d, d], self.sigma_floor**2)
            else:
                var_e = elites.var(axis=0) + self.sigma_floor**2
                C = (1 - self.alpha_c) * C + self.alpha_c * var_e
            m = (1 - self.alpha_m) * m + self.alpha_m * m_e

        return best_x, best_score, best_extra


class DE:
    """
    差分进化（Differential Evolution）—— rand/1/bin
    - 全局搜索强，易跳出局部
    """
    def __init__(self, f_eval, bounds, pop_size=160, iters=200,
                 F_min=0.5, F_max=1.0, CR=0.9, seed=2025, init_hint: Optional[np.ndarray] = None):
        self.f_eval = f_eval
        self.bounds = bounds
        self.pop_size = pop_size
        self.iters = iters
        self.F_min = F_min
        self.F_max = F_max
        self.CR = CR
        self.rng = np.random.default_rng(seed)
        self.init_hint = init_hint

    def _clip_vec(self, x):
        for j, (lo, hi) in enumerate(self.bounds):
            if j % 4 == 0:
                x[j] = x[j] % (2.0 * math.pi)
            else:
                x[j] = lo if x[j] < lo else (hi if x[j] > hi else x[j])
        return x

    def _init_pop(self):
        D = len(self.bounds)
        X = np.zeros((self.pop_size, D), dtype=float)
        for j, (lo, hi) in enumerate(self.bounds):
            X[:, j] = self.rng.uniform(lo, hi, size=self.pop_size)
        if self.init_hint is not None and len(self.init_hint) == D:
            X[0] = self._clip_vec(self.init_hint.copy())
        return X

    def optimize(self):
        X = self._init_pop()
        scores = np.zeros(self.pop_size, dtype=float)
        extras = [None] * self.pop_size
        for i in range(self.pop_size):
            s, extra = self.f_eval(self._clip_vec(X[i].copy()))
            scores[i], extras[i] = s, extra
        best_idx = int(np.argmax(scores))
        best_x, best_s, best_extra = X[best_idx].copy(), float(scores[best_idx]), extras[best_idx]

        D = len(self.bounds)
        for _ in range(self.iters):
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = self.rng.choice(idxs, size=3, replace=False)
                F = self.rng.uniform(self.F_min, self.F_max)
                v = X[r1] + F * (X[r2] - X[r3])  # 变异
                # 交叉
                u = X[i].copy()
                j_rand = self.rng.integers(0, D)
                for j in range(D):
                    if self.rng.random() < self.CR or j == j_rand:
                        u[j] = v[j]
                u = self._clip_vec(u)
                # 选择
                s_u, extra_u = self.f_eval(u)
                if s_u >= scores[i]:
                    X[i], scores[i], extras[i] = u, s_u, extra_u
                    if s_u > best_s:
                        best_x, best_s, best_extra = u.copy(), float(s_u), extra_u

        return best_x, best_s, best_extra

# =========================
# 六、统一求解接口（Q4）
# =========================

def solve_q4(strategy: str = "two_stage",        # 'L0' | 'L1' | 'two_stage'
             optimizer: str = "CEM",             # 'PSO' | 'CEM' | 'DE'
             # 数值参数
             dt_L0: float = 0.01,
             N_ANG: int = 64, N_Z: int = 12, INCLUDE_SIDE: bool = True, dt_L1: float = 0.02,
             # 优化器参数（阶段1）
             swarm_size: int = 128, iters: int = 220, seed: int = 2025,  # 供 PSO/DE
             cem_pop: int = 240, cem_iters: int = 120, elite_frac: float = 0.15, use_full_cov: bool = False,
             # 两阶段阶段2（L1 微调）
             stage2_swarm: int = 80, stage2_iters: int = 90,            # PSO/DE
             stage2_cem_pop: int = 160, stage2_cem_iters: int = 100,     # CEM
             # 可选初始化提示：长度12 (θ1,v1,t1,τ1, θ2,v2,t2,τ2, θ3,v3,t3,τ3)
             init_hint: Optional[Tuple[float, ...]] = None
             ) -> Dict[str, Any]:
    """
    返回：参数、起爆信息、并集遮蔽总时长与区间
    """

    # 12 维边界（每4维一组）
    bounds = []
    for _ in range(3):
        bounds += [
            (0.0, 2.0 * math.pi),  # theta
            (70.0, 140.0),         # v
            (0.0, 60.0),           # t_drop
            (0.2, 12.0),           # tau
        ]

    # L1 采样点
    PTS = None
    if strategy in ("L1", "two_stage"):
        PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)

    # 评价函数（把12维拆成三组）
    def f_eval_L0(vec):
        ths = [vec[0], vec[4], vec[8]]
        vs  = [vec[1], vec[5], vec[9]]
        tds = [vec[2], vec[6], vec[10]]
        taus= [vec[3], vec[7], vec[11]]
        cover, intervals, info = eval_cover_time_q4_L0(ths, vs, tds, taus, dt=dt_L0)
        return cover, {"intervals": intervals, "info": info, "theta": ths, "v": vs}

    def f_eval_L1(vec):
        ths = [vec[0], vec[4], vec[8]]
        vs  = [vec[1], vec[5], vec[9]]
        tds = [vec[2], vec[6], vec[10]]
        taus= [vec[3], vec[7], vec[11]]
        cover, intervals, info = eval_cover_time_q4_L1(ths, vs, tds, taus, PTS=PTS, dt=dt_L1)
        return cover, {"intervals": intervals, "info": info, "theta": ths, "v": vs}

    # 初值
    init_vec = None
    if init_hint is not None and len(init_hint) == 12:
        init_vec = np.array(list(init_hint), dtype=float)
        for j, (lo, hi) in enumerate(bounds):
            if j % 4 == 0:  # theta
                init_vec[j] = init_vec[j] % (2.0 * math.pi)
            else:
                init_vec[j] = clip(init_vec[j], lo, hi)

    # 选择优化器构造器
    def build_opt(f_eval, bounds, phase=1, ref_x=None):
        opt = optimizer.upper()
        if opt == "CEM":
            if phase == 1:
                return CEM(f_eval, bounds, pop_size=cem_pop, iters=cem_iters,
                           elite_frac=elite_frac, use_full_cov=use_full_cov,
                           seed=seed, init_hint=init_vec)
            else:
                ih = ref_x if ref_x is not None else init_vec
                return CEM(f_eval, bounds, pop_size=stage2_cem_pop, iters=stage2_cem_iters,
                           elite_frac=elite_frac, use_full_cov=True,  # 精修阶段建议开全协方差
                           seed=seed+1, init_hint=ih)
        elif opt == "DE":
            if phase == 1:
                return DE(f_eval, bounds, pop_size=swarm_size, iters=iters, seed=seed, init_hint=init_vec)
            else:
                ih = ref_x if ref_x is not None else init_vec
                return DE(f_eval, bounds, pop_size=stage2_swarm, iters=stage2_iters, seed=seed+1, init_hint=ih)
        else:  # PSO
            if phase == 1:
                return PSO(f_eval, bounds, swarm_size=swarm_size, iters=iters, seed=seed, init_hint=init_vec)
            else:
                ih = ref_x if ref_x is not None else init_vec
                return PSO(f_eval, bounds, swarm_size=stage2_swarm, iters=stage2_iters, seed=seed+1, init_hint=ih)

    # 求解
    if strategy == "L0":
        opt1 = build_opt(f_eval_L0, bounds, phase=1)
        best_x, best_cover, extra = opt1.optimize()
        eval_used = "L0"
    elif strategy == "L1":
        opt1 = build_opt(f_eval_L1, bounds, phase=1)
        best_x, best_cover, extra = opt1.optimize()
        eval_used = "L1"
    elif strategy == "two_stage":
        # 阶段1：L0 全局
        opt1 = build_opt(f_eval_L0, bounds, phase=1)
        x1, cover1, extra1 = opt1.optimize()
        # 阶段2：L1 精修
        opt2 = build_opt(f_eval_L1, bounds, phase=2, ref_x=x1)
        best_x, best_cover, extra = opt2.optimize()
        eval_used = "two_stage"
    else:
        raise ValueError("strategy 必须为 'L0'、'L1' 或 'two_stage'")

    # 整理输出
    ths = [float(extra["theta"][0]), float(extra["theta"][1]), float(extra["theta"][2])]
    vs  = [float(extra["v"][0]), float(extra["v"][1]), float(extra["v"][2])]
    info = extra["info"]
    intervals = extra["intervals"]

    out = {
        "strategy": eval_used,
        "theta_deg": [round(math.degrees(t), 3) for t in ths],
        "v_u_mps": [round(v, 3) for v in vs],
        "bursts": info["bursts"],  # 按起爆时刻排序
        "cover_total_s": best_cover,
        "cover_intervals_s": [(round(a, 3), round(b, 3)) for (a, b) in intervals],
        "config": {
            "strategy": strategy, "optimizer": optimizer,
            "dt_L0": dt_L0, "dt_L1": dt_L1, "N_ANG": N_ANG, "N_Z": N_Z, "INCLUDE_SIDE": INCLUDE_SIDE,
            "swarm_size": swarm_size, "iters": iters,
            "stage2_swarm": stage2_swarm, "stage2_iters": stage2_iters,
            "cem_pop": cem_pop, "cem_iters": cem_iters, "elite_frac": elite_frac, "use_full_cov": use_full_cov,
            "stage2_cem_pop": stage2_cem_pop, "stage2_cem_iters": stage2_cem_iters
        }
    }
    return out

# =========================
# 七、几何引导初始化（早/中/晚）
# =========================

def make_q4_geometry_seed(
    fracs=(0.15, 0.50, 0.82),   # 三段相对时刻：早/中/晚（相对于 min(60, T_HIT-2)）
    alphas=(0.82, 0.76, 0.72),  # 视线线段内比例点：越大越靠近导弹端
    tau_seeds=(1.0, 1.4, 2.2)   # 三枚引信初始值（秒）
) -> Tuple[float, ...]:
    """
    返回长度12的初始化向量：
      (θ1, v1, t1, τ1,  θ2, v2, t2, τ2,  θ3, v3, t3, τ3)
    几何策略：
      1) 选 t_b^* = frac * min(60, T_HIT-2)
      2) 视线点 Y* = P + alpha*(m(t_b^*)-P)
      3) 航向指向 Y*_xy，速度 v = clip(距离(u0_xy, Y*_xy) / t_b^*, [70,140])
      4) τ 取给定种子（若导致落地则缩短），t_drop = t_b^* - τ（裁剪到[0,60]）
    """
    seeds = []
    U0s = [U0_FY1, U0_FY2, U0_FY3]
    for i in range(3):
        u0 = U0s[i]
        frac = fracs[i]
        alpha = alphas[i]
        tau0 = clip(tau_seeds[i], 0.2, 12.0)

        t_b = frac * min(60.0, T_HIT - 2.0)
        t_b = clip(t_b, 0.5, 59.5)

        m_tb = missile_pos(M0, t_b)
        Y = P_TARGET + alpha * (m_tb - P_TARGET)

        dx, dy = Y[0] - u0[0], Y[1] - u0[1]
        theta = math.atan2(dy, dx)
        D_xy = math.hypot(dx, dy)
        v = clip(D_xy / t_b, 70.0, 140.0)

        tau = tau0
        z_burst = u0[2] - 0.5 * g * tau * tau
        if z_burst <= 0.0:
            tau = max(0.2, math.sqrt(2.0 * (u0[2] - 1.0) / g))  # 缩短到刚好>0
        t_drop = clip(t_b - tau, 0.0, 60.0)

        seeds += [theta % (2.0 * math.pi), v, t_drop, tau]

    return tuple(seeds)

# =========================
# 八、单枚贡献诊断（可选）
# =========================

def _single_cloud_cover_L0(u0, theta, v, t_drop, tau, dt=0.01):
    """仅开启一枚云团（L0）时的遮蔽时长与区间"""
    b = _valid_burst(u0, theta, v, t_drop, tau)
    if b is None:
        return 0.0, []
    t_drop, tau, t_burst, s_burst = b
    t0 = t_burst
    t1 = min(t_burst + T_EFFECT, T_HIT)
    covered, intervals, in_seg, seg_start = 0.0, [], False, None
    t = t0
    while t <= t1 + 1e-12:
        if covered_L0_at_time(M0, P_TARGET, s_burst, t_burst, t):
            if not in_seg:
                in_seg, seg_start = True, t
            covered += dt
        else:
            if in_seg:
                in_seg = False
                intervals.append((seg_start, t))
        t += dt
    if in_seg:
        intervals.append((seg_start, t1))
    return covered, intervals

def _single_cloud_cover_L1(u0, theta, v, t_drop, tau, PTS, dt=0.02):
    """仅开启一枚云团（L1）"""
    b = _valid_burst(u0, theta, v, t_drop, tau)
    if b is None:
        return 0.0, []
    t_drop, tau, t_burst, s_burst = b
    t0 = t_burst
    t1 = min(t_burst + T_EFFECT, T_HIT)
    covered, intervals, in_seg, seg_start = 0.0, [], False, None
    t = t0
    while t <= t1 + 1e-12:
        if covered_L1_at_time_vectorized(M0, s_burst, t_burst, t, PTS):
            if not in_seg:
                in_seg, seg_start = True, t
            covered += dt
        else:
            if in_seg:
                in_seg = False
                intervals.append((seg_start, t))
        t += dt
    if in_seg:
        intervals.append((seg_start, t1))
    return covered, intervals

def per_cloud_coverage_report(strategy: str,
                              thetas: List[float], vs: List[float],
                              drops: List[float], taus: List[float],
                              N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
                              dt_L0=0.01, dt_L1=0.02) -> Dict[str, Any]:
    """
    分别评估 FY1/FY2/FY3 单独起作用时的遮蔽时长（定位贡献薄弱的那一枚）
    返回：{'FY1': {...}, 'FY2': {...}, 'FY3': {...}}
    """
    res = {}
    PTS = None
    if strategy in ("L1", "two_stage"):
        PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)

    for i, uav in enumerate(UAVS):
        u0 = uav["U0"]
        th, v, td, ta = thetas[i], vs[i], drops[i], taus[i]
        if strategy == "L0":
            cov, ints = _single_cloud_cover_L0(u0, th, v, td, ta, dt=dt_L0)
        else:
            cov, ints = _single_cloud_cover_L1(u0, th, v, td, ta, PTS=PTS, dt=dt_L1)
        res[uav["name"]] = {
            "cover_single_s": cov,
            "intervals_s": [(round(a, 3), round(b, 3)) for (a, b) in ints]
        }
    return res

# =========================
# 九、示例调用（直接可跑）
# =========================
if __name__ == "__main__":
    # 1) 几何引导的初值（让三枚对齐早/中/晚三段）
    seed_vec = make_q4_geometry_seed(
        fracs=(0.15, 0.50, 0.82),
        alphas=(0.82, 0.76, 0.72),
        tau_seeds=(1.0, 1.4, 2.2)
    )

    # 2) 推荐：两阶段（L0→L1），优化器用 CEM（阶段2 开启全协方差）
    ans = solve_q4(
        strategy="two_stage",
        optimizer="CEM",
        dt_L0=0.01,
        N_ANG=64, N_Z=12, INCLUDE_SIDE=True, dt_L1=0.02,
        cem_pop=240, cem_iters=120, elite_frac=0.15, use_full_cov=False,   # 阶段1(L0)
        stage2_cem_pop=160, stage2_cem_iters=100,                           # 阶段2(L1)
        swarm_size=128, iters=220, seed=2025,
        init_hint=seed_vec
    )
    print("[Q4 | two_stage + CEM] 最优：")
    for k, v in ans.items():
        print(f"  {k}: {v}")

    # —— 可选：只用 L0（最快）
    # ans = solve_q4(strategy="L0", optimizer="DE", dt_L0=0.01,
    #                swarm_size=160, iters=200, seed=2025, init_hint=seed_vec)
    # print("\n[Q4 | L0 + DE] 最优："); [print(f"  {k}: {v}") for k, v in ans.items()]

    # 3) 单枚贡献诊断（针对上面结果）
    ths_deg = ans["theta_deg"]
    ths = [math.radians(d) for d in ths_deg]
    vs  = ans["v_u_mps"]
    # 将 bursts 中的参数按 UAV 名称归位
    drops_map, taus_map = {}, {}
    for b in ans["bursts"]:
        drops_map[b["uav"]] = b["t_drop"]
        taus_map[b["uav"]]  = b["tau"]
    drops = [drops_map["FY1"], drops_map["FY2"], drops_map["FY3"]]
    taus  = [taus_map["FY1"],  taus_map["FY2"],  taus_map["FY3"]]

    report = per_cloud_coverage_report("two_stage", ths, vs, drops, taus,
                                       N_ANG=64, N_Z=12, INCLUDE_SIDE=True,
                                       dt_L0=0.01, dt_L1=0.02)
    print("\n[单枚贡献诊断 | two_stage]")
    for uav, info in report.items():
        print(f"  {uav}: 单独覆盖 {info['cover_single_s']:.3f} s, 区间 {info['intervals_s']}")
