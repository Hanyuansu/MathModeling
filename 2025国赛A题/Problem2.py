# -*- coding: utf-8 -*-
"""
CUMCM 2025 A题 —— 第2问：单机单弹最优 (L1 判定 + PSO 优化)
- L1：对目标圆柱表面采样（上下圆面 + 侧壁多层），“任一点被遮蔽即算遮蔽”
- 判定全向量化：一次性计算所有采样点到“导弹→采样点”的线段距离，极快
- 优化器：自实现粒子群 PSO（与 L0 版本完全兼容）

【如何提速 or 提精度】
  1) 采样密度：N_ANG（环向点数）、N_Z（侧壁高度层数）
  2) 时间步长 DT
  3) PSO 群体规模与迭代次数 SWARM_SIZE / ITERS
"""

import math
import numpy as np
from typing import Tuple, List, Optional

# =========================
# 一、题面常数与场景参数
# =========================

g = 9.81                 # 重力加速度 (m/s^2)
VM = 300.0               # 导弹速度 (m/s)，匀速直线指向假目标(原点)
V_SINK = 3.0             # 云团下沉速度 (m/s)
R_SMOKE = 10.0           # 云团有效半径 (m)
T_EFFECT = 20.0          # 起爆后有效时间 (s)

# 目标圆柱：半径7m，高10m；下底面圆心在(0,200,0)
R_TAR, H_TAR = 7.0, 10.0
CYL_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)  # (cx, cy, z0)

# 导弹与无人机初始状态（以第1问场景为例）
M0 = np.array([20000.0, 0.0, 2000.0], dtype=float)  # M1 初始坐标
U0 = np.array([17800.0, 0.0, 1800.0], dtype=float)  # FY1 初始坐标（等高度飞行）

def missile_hit_time(m0: np.ndarray) -> float:
    """导弹直线等速命中原点的时刻 T_hit = ||m0|| / VM。用于裁剪积分上限"""
    return float(np.linalg.norm(m0) / VM)

T_HIT = missile_hit_time(M0)

# =========================
# 二、模型函数（运动学 + L1几何）
# =========================

def unit(v: np.ndarray) -> np.ndarray:
    """单位向量：避免除零"""
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v

def missile_pos(m0: np.ndarray, t: float) -> np.ndarray:
    """导弹位置：匀速直线指向原点"""
    d = unit(-m0)
    return m0 + VM * d * t

def uav_pos(u0: np.ndarray, theta: float, v_u: float, t: float) -> np.ndarray:
    """无人机：等高度直线匀速；theta 为航向角（弧度）"""
    hx, hy = math.cos(theta), math.sin(theta)
    return np.array([u0[0] + v_u * hx * t, u0[1] + v_u * hy * t, u0[2]], dtype=float)

def burst_point(u0: np.ndarray, theta: float, v_u: float, t_drop: float, tau: float) -> np.ndarray:
    """
    起爆点（球心初值）：
      r_drop = uav_pos(u0, theta, v_u, t_drop)
      s_burst = r_drop + [v_u*hx*tau, v_u*hy*tau, -0.5*g*tau^2]
    """
    hx, hy = math.cos(theta), math.sin(theta)
    r_drop = uav_pos(u0, theta, v_u, t_drop)
    horiz = np.array([v_u * hx * tau, v_u * hy * tau, 0.0], dtype=float)
    vert  = np.array([0.0, 0.0, -0.5 * g * tau * tau], dtype=float)
    return r_drop + horiz + vert

def smoke_center_after_burst(s_burst: np.ndarray, t: float, t_burst: float) -> np.ndarray:
    """起爆后云团球心：3 m/s 匀速下沉"""
    dz = -V_SINK * max(0.0, t - t_burst)
    return s_burst + np.array([0.0, 0.0, dz], dtype=float)

# ---------- 圆柱表面采样（上下圆面 + 侧壁） ----------
def cyl_points_top_bottom(N_ang: int = 48) -> np.ndarray:
    """上下圆面：各 N_ang 个点，共 2*N_ang"""
    cx, cy, _ = CYL_CENTER
    out = []
    for z in (0.0, H_TAR):
        for k in range(N_ang):
            ang = 2.0 * math.pi * k / N_ang
            x = cx + R_TAR * math.cos(ang)
            y = cy + R_TAR * math.sin(ang)
            out.append((x, y, z))
    return np.array(out, dtype=float)

def cyl_points_side(N_ang: int = 48, N_z: int = 9) -> np.ndarray:
    """侧壁：沿高度方向 N_z 层，每层 N_ang 个点"""
    cx, cy, _ = CYL_CENTER
    zs = np.linspace(0.0, H_TAR, N_z)
    out = []
    for z in zs:
        for k in range(N_ang):
            ang = 2.0 * math.pi * k / N_ang
            x = cx + R_TAR * math.cos(ang)
            y = cy + R_TAR * math.sin(ang)
            out.append((x, y, z))
    return np.array(out, dtype=float)

def build_cylinder_samples(N_ang=48, N_z=9, include_side=True) -> np.ndarray:
    """
    生成 L1 判定用的采样点集合
    - include_side=True 时：上下圆面 + 侧壁
    - include_side=False 时：仅上下圆面（更快，略保守）
    """
    pts = [cyl_points_top_bottom(N_ang)]
    if include_side:
        pts.append(cyl_points_side(N_ang, N_z))
    return np.concatenate(pts, axis=0)

# ---------- 向量化的 L1 单时刻遮蔽判定 ----------
def covered_L1_at_time_vectorized(m0: np.ndarray,
                                  s_burst: np.ndarray,
                                  t_burst: float,
                                  t: float,
                                  PTS: np.ndarray) -> bool:
    """
    判定：时刻 t，是否存在“圆柱采样点 p”，使得
           距离(球心 s(t), 线段 p→m(t)) <= R_SMOKE
    向量化实现：对所有 p ∈ PTS 并行计算
    """
    m_t = missile_pos(m0, t)           # shape (3,)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)  # shape (3,)

    # v = m_t - p, w = s_t - p
    v = m_t - PTS                      # shape (N, 3)
    w = s_t - PTS                      # shape (N, 3)
    vv = np.sum(v * v, axis=1)         # (N,)

    # 线段参数 alpha = clip( (w·v)/(v·v), 0, 1 )
    # 避免除0：vv>0 几乎恒真（导弹不可能恰好在采样点处）
    alpha = np.divide(np.sum(w * v, axis=1), vv, out=np.zeros_like(vv), where=vv>0.0)
    alpha = np.clip(alpha, 0.0, 1.0)   # (N,)

    # 最近点 Y = p + alpha * v
    Y = PTS + alpha[:, None] * v       # (N,3)
    dist = np.linalg.norm(s_t - Y, axis=1)  # (N,)
    return bool(np.any(dist <= R_SMOKE))

# =========================
# 三、目标函数（L1）
# =========================

def clip_bounds(x, lo, hi):
    """数值裁剪到[lo, hi]"""
    return lo if x < lo else (hi if x > hi else x)

def eval_cover_time_L1(theta: float, v_u: float, t_drop: float, tau: float,
                       PTS: np.ndarray,
                       dt: float = 0.02) -> Tuple[float, List[Tuple[float, float]]]:
    """
    评价函数（L1）：给定(θ, v, t_drop, tau)返回遮蔽总时长与区间
    - PTS：圆柱采样点数组（预先 build_cylinder_samples）
    - 时间积分在 [t_burst, min(t_burst+20, T_HIT)] 上进行
    """
    # 基本边界
    theta = theta % (2.0 * math.pi)
    v_u   = clip_bounds(v_u,   70.0, 140.0)
    t_drop= clip_bounds(t_drop,0.0,  60.0)
    tau   = clip_bounds(tau,   0.2,  12.0)

    # 起爆信息
    t_burst = t_drop + tau
    if t_burst >= T_HIT:
        return 0.0, []  # 导弹已命中/即将命中，来不及遮蔽

    s_burst = burst_point(U0, theta, v_u, t_drop, tau)
    if s_burst[2] <= 0.0:
        return 0.0, []  # 起爆点在地面/以下，判无效

    t0 = t_burst
    t1 = min(t_burst + T_EFFECT, T_HIT)
    if t1 <= t0:
        return 0.0, []

    # 0/1 指示积分
    covered = 0.0
    intervals: List[Tuple[float, float]] = []
    in_seg = False
    seg_start = None

    t = t0
    while t <= t1 + 1e-12:
        flag = covered_L1_at_time_vectorized(M0, s_burst, t_burst, t, PTS)
        if flag and not in_seg:
            in_seg = True
            seg_start = t
        if (not flag) and in_seg:
            in_seg = False
            intervals.append((seg_start, t))
        if flag:
            covered += dt
        t += dt

    if in_seg:
        intervals.append((seg_start, t1))

    return covered, intervals

# =========================
# 四、PSO（与 L0 版本一致，可复用）
# =========================

class PSO:
    """
    粒子群优化（连续变量）
    变量：x = [theta, v_u, t_drop, tau]
    目标：最大化遮蔽时长（实现上用 -cover 作为损失最小化）
    """
    def __init__(self,
                 f_eval,             # f_eval(theta, v, t_drop, tau) -> (cover, intervals)
                 bounds,             # [(lo, hi)]*4
                 swarm_size=64,
                 iters=120,
                 inertia_w=0.72,
                 c1=1.49,
                 c2=1.49,
                 seed=2025,
                 init_hint: Optional[np.ndarray] = None):
        self.f_eval = f_eval
        self.bounds = bounds
        self.swarm_size = swarm_size
        self.iters = iters
        self.w = inertia_w
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.default_rng(seed)
        self.init_hint = init_hint  # 可注入 L0 的好解作为群体里的一个粒子

    def _init_swarm(self):
        D = len(self.bounds)
        X = np.zeros((self.swarm_size, D), dtype=float)
        V = np.zeros((self.swarm_size, D), dtype=float)
        for j, (lo, hi) in enumerate(self.bounds):
            X[:, j] = self.rng.uniform(lo, hi, size=self.swarm_size)
            span = hi - lo
            V[:, j] = self.rng.uniform(-0.1 * span, 0.1 * span, size=self.swarm_size)

        # 把 init_hint 塞进第0号粒子（若提供）
        if self.init_hint is not None and len(self.init_hint) == D:
            X[0] = self.init_hint.copy()
            V[0] = 0.0

        pbest_X = X.copy()
        pbest_val = np.full(self.swarm_size, np.inf, dtype=float)
        gbest_x = None
        gbest_val = np.inf
        return X, V, pbest_X, pbest_val, gbest_x, gbest_val

    def _loss(self, x):
        theta = float(x[0]) % (2.0 * math.pi)
        v_u   = clip_bounds(float(x[1]), self.bounds[1][0], self.bounds[1][1])
        t_drop= clip_bounds(float(x[2]), self.bounds[2][0], self.bounds[2][1])
        tau   = clip_bounds(float(x[3]), self.bounds[3][0], self.bounds[3][1])

        cover, intervals = self.f_eval(theta, v_u, t_drop, tau)
        loss = -cover
        return loss, cover, {"intervals": intervals, "theta": theta, "v": v_u, "t_drop": t_drop, "tau": tau}

    def optimize(self):
        X, V, pbest_X, pbest_val, gbest_x, gbest_val = self._init_swarm()

        # 初评估
        for i in range(self.swarm_size):
            loss_i, cover_i, info_i = self._loss(X[i])
            pbest_X[i] = X[i].copy()
            pbest_val[i] = loss_i
            if loss_i < gbest_val:
                gbest_val = loss_i
                gbest_x = X[i].copy()
                gbest_info = info_i

        # 迭代
        for it in range(self.iters):
            w = self.w
            for i in range(self.swarm_size):
                r1 = self.rng.random(len(self.bounds))
                r2 = self.rng.random(len(self.bounds))
                V[i] = w * V[i] + self.c1 * r1 * (pbest_X[i] - X[i]) + self.c2 * r2 * (gbest_x - X[i])
                X[i] = X[i] + V[i]
                # 边界处理
                for j, (lo, hi) in enumerate(self.bounds):
                    if j == 0:  # theta
                        X[i, j] = X[i, j] % (2.0 * math.pi)
                    else:
                        if X[i, j] < lo or X[i, j] > hi:
                            X[i, j] = clip_bounds(X[i, j], lo, hi)
                            V[i, j] *= -0.5
                # 评估
                loss_i, cover_i, info_i = self._loss(X[i])
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
# 五、主入口（L1 + PSO）
# =========================

def solve_q2_with_pso_L1(
    N_ANG: int = 48,
    N_Z: int   = 9,
    INCLUDE_SIDE: bool = True,
    DT: float = 0.02,
    SWARM_SIZE: int = 64,
    ITERS: int = 120,
    init_hint: Optional[Tuple[float, float, float, float]] = None  # (theta, v, t_drop, tau)
):
    """
    运行 L1 + PSO 求解 θ、v、t_drop、tau 的最优
    - 可传入 L0 求得的 init_hint 作为群体初值的一员
    """
    # 预构建圆柱采样点（一次性）
    PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)

    def f_eval(theta, v, t_drop, tau):
        cover, intervals = eval_cover_time_L1(theta, v, t_drop, tau, PTS=PTS, dt=DT)
        return cover, intervals

    bounds = [
        (0.0, 2.0 * math.pi),  # theta
        (70.0, 140.0),         # v_u
        (0.0, 60.0),           # t_drop
        (0.2, 12.0),           # tau
    ]

    # 把 init_hint 限制到边界范围，并做 θ 2π 包裹
    init_vec = None
    if init_hint is not None:
        th, vv, td, ta = init_hint
        init_vec = np.array([
            float(th) % (2.0 * math.pi),
            clip_bounds(float(vv), bounds[1][0], bounds[1][1]),
            clip_bounds(float(td), bounds[2][0], bounds[2][1]),
            clip_bounds(float(ta), bounds[3][0], bounds[3][1]),
        ], dtype=float)

    pso = PSO(f_eval, bounds, swarm_size=SWARM_SIZE, iters=ITERS,
              inertia_w=0.72, c1=1.49, c2=1.49, seed=2025, init_hint=init_vec)
    best_x, best_cover, info = pso.optimize()

    theta = info["theta"]
    v = info["v"]
    t_drop = info["t_drop"]
    tau = info["tau"]
    t_burst = t_drop + tau
    s_burst = burst_point(U0, theta, v, t_drop, tau)

    result = {
        "theta_deg": math.degrees(theta),
        "v_u_mps": v,
        "t_drop_s": t_drop,
        "tau_s": tau,
        "t_burst_s": t_burst,
        "burst_point_m": (float(s_burst[0]), float(s_burst[1]), float(s_burst[2])),
        "cover_total_s": best_cover,
        "cover_intervals_s": [(round(a, 3), round(b, 3)) for (a, b) in info["intervals"]],
        "config": {"N_ANG": N_ANG, "N_Z": N_Z, "INCLUDE_SIDE": INCLUDE_SIDE, "DT": DT,
                   "SWARM_SIZE": SWARM_SIZE, "ITERS": ITERS}
    }
    return result

# 说明：为避免你“复制即运行”就长时间计算，这里不主动调用。
# 如果需要执行，请在你的脚本里这样用（示例）：
if __name__ == "__main__":
    # 可选：把 L0 找到的好解作为初始化提示
    init_hint = (math.radians(6.9), 99, 0, 1)  # (theta, v, t_drop, tau) 例子
    ans = solve_q2_with_pso_L1(N_ANG=48, N_Z=9, INCLUDE_SIDE=True, DT=0.02,
                               SWARM_SIZE=64, ITERS=120, init_hint=init_hint)
    print("[Q2 | L1+PSO] 最优解：")
    for k, v in ans.items():
        print(f"  {k}: {v}")
