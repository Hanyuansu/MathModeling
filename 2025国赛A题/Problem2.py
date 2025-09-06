# -*- coding: utf-8 -*-
"""
CUMCM 2025 A题 —— 第2问：单机单弹最优（L0 / L1 / two_stage 通用版）
- L0：以圆柱中心点代表视轴，球-线段相交；极快
- L1：对圆柱表面采样（上下圆面 + 侧壁），向量化判定；稳
- two_stage：先 L0 做全局搜索，再以 L1 小群体微调（推荐）

返回字段统一：
  {
    "strategy": ...,                 # 使用的策略
    "theta_deg": ..., "v_u_mps": ...,
    "t_drop_s": ..., "tau_s": ...,
    "t_burst_s": ..., "burst_point_m": (x,y,z),
    "cover_total_s": ...,            # 遮蔽总时长
    "cover_intervals_s": [...],      # 遮蔽时间区间列表（精确）
    "cover_intervals_pretty": [...], # 仅展示用的四舍五入版本
    "config": {...}                  # 本次求解的配置
  }
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import os
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
P_TARGET   = np.array([0.0, 200.0, 5.0], dtype=float)  # L0 代表点（圆柱几何中心）

# 导弹与无人机初始状态（以第1问场景为例）
M0 = np.array([20000.0, 0.0, 2000.0], dtype=float)  # M1 初始坐标
U0 = np.array([17800.0, 0.0, 1800.0], dtype=float)  # FY1 初始坐标（等高度飞行）

def missile_hit_time(m0: np.ndarray) -> float:
    """导弹直线等速命中原点的时刻 T_hit = ||m0|| / VM。用于裁剪积分上限"""
    return float(np.linalg.norm(m0) / VM)

# =========================
# 二、通用运动学/几何
# =========================

def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v

def missile_pos(m0: np.ndarray, t: float) -> np.ndarray:
    d = unit(-m0)
    return m0 + VM * d * t

def uav_pos(u0: np.ndarray, theta: float, v_u: float, t: float) -> np.ndarray:
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
    dz = -V_SINK * max(0.0, t - t_burst)
    return s_burst + np.array([0.0, 0.0, dz], dtype=float)

def point_to_segment_dist(P: np.ndarray, Q: np.ndarray, X: np.ndarray) -> float:
    """点X到线段PQ的最小距离"""
    v = Q - P
    vv = float(np.dot(v, v))
    if vv == 0.0:
        return float(np.linalg.norm(X - P))
    a = float(np.dot(X - P, v) / vv)
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    Y = P + a * v
    return float(np.linalg.norm(X - Y))

def clip_bounds(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

# =========================
# 三、区间规范化工具（新增）
# =========================

def _canonize_intervals(intervals, t0, t1, dt, eps=1e-9):
    """
    规范化区间到 [t0,t1]，并吸附到时间网格，去重叠/毛刺：
      - 裁剪边界
      - snap 到 t0 + k*dt
      - 合并“几乎相接/轻微重叠”的区间
      - 返回升序、互不重叠的列表
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda ab: ab[0])

    def snap(x):
        k = round((x - t0) / dt)
        return t0 + k * dt

    out = []
    for a, b in intervals:
        a = max(t0, min(t1, a))
        b = max(t0, min(t1, b))
        a = snap(a); b = snap(b)
        if b <= a + eps:
            continue
        if not out:
            out.append((a, b))
        else:
            pa, pb = out[-1]
            if a <= pb + eps:
                out[-1] = (pa, max(pb, b))
            else:
                out.append((a, b))
    return out

# =========================
# 四、L0 / L1 单时刻判定
# =========================

def covered_L0_at_time(m0, p_target, s_burst, t_burst, t) -> bool:
    """L0：以圆柱中心点为代表视轴，球-线段相交判定"""
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    return (point_to_segment_dist(p_target, m_t, s_t) <= R_SMOKE)

# ---- 圆柱表面采样（上下圆面 + 侧壁） ----
def cyl_points_top_bottom(N_ang: int = 48) -> np.ndarray:
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
    pts = [cyl_points_top_bottom(N_ang)]
    if include_side:
        pts.append(cyl_points_side(N_ang, N_z))
    return np.concatenate(pts, axis=0)

def covered_L1_at_time_vectorized(m0: np.ndarray,
                                  s_burst: np.ndarray,
                                  t_burst: float,
                                  t: float,
                                  PTS: np.ndarray) -> bool:
    """L1：是否存在采样点被遮蔽（向量化）"""
    m_t = missile_pos(m0, t)
    s_t = smoke_center_after_burst(s_burst, t, t_burst)
    v = m_t - PTS                      # (N,3)
    w = s_t - PTS                      # (N,3)
    vv = np.sum(v * v, axis=1)         # (N,)
    alpha = np.divide(np.sum(w * v, axis=1), vv, out=np.zeros_like(vv), where=vv > 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    Y = PTS + alpha[:, None] * v
    dist = np.linalg.norm(s_t - Y, axis=1)
    return bool(np.any(dist <= R_SMOKE))

# =========================
# 五、评价函数（L0 / L1）——已加“区间规范化”与动态 T_HIT
# =========================

def eval_cover_time_L0(theta: float, v_u: float, t_drop: float, tau: float,
                       dt: float = 0.01) -> Tuple[float, List[Tuple[float, float]]]:
    """给定参数，返回 L0 遮蔽总时长与区间（已规范化）"""
    theta = theta % (2.0 * math.pi)
    v_u   = clip_bounds(v_u,   70.0, 140.0)
    t_drop= clip_bounds(t_drop,0.0,  60.0)
    tau   = clip_bounds(tau,   0.2,  12.0)

    T_HIT_loc = missile_hit_time(M0)
    t_burst = t_drop + tau
    if t_burst >= T_HIT_loc:
        return 0.0, []

    s_burst = burst_point(U0, theta, v_u, t_drop, tau)
    if s_burst[2] <= 0.0:
        return 0.0, []

    t0 = t_burst
    t1 = min(t_burst + T_EFFECT, T_HIT_loc)
    if t1 <= t0:
        return 0.0, []

    intervals: List[Tuple[float, float]] = []
    in_seg = False
    seg_start = None

    t = t0
    while t <= t1 + 1e-12:
        flag = covered_L0_at_time(M0, P_TARGET, s_burst, t_burst, t)
        if flag and not in_seg:
            in_seg, seg_start = True, t
        if (not flag) and in_seg:
            in_seg = False
            intervals.append((seg_start, t))
        t += dt
    if in_seg:
        intervals.append((seg_start, t1))

    # 规范化 + 覆盖由区间求和
    intervals = _canonize_intervals(intervals, t0, t1, dt)
    covered = sum(b - a for (a, b) in intervals)
    return covered, intervals

def eval_cover_time_L1(theta: float, v_u: float, t_drop: float, tau: float,
                       PTS: np.ndarray, dt: float = 0.02) -> Tuple[float, List[Tuple[float, float]]]:
    """给定参数，返回 L1 遮蔽总时长与区间（已规范化）"""
    theta = theta % (2.0 * math.pi)
    v_u   = clip_bounds(v_u,   70.0, 140.0)
    t_drop= clip_bounds(t_drop,0.0,  60.0)
    tau   = clip_bounds(tau,   0.2,  12.0)

    T_HIT_loc = missile_hit_time(M0)
    t_burst = t_drop + tau
    if t_burst >= T_HIT_loc:
        return 0.0, []

    s_burst = burst_point(U0, theta, v_u, t_drop, tau)
    if s_burst[2] <= 0.0:
        return 0.0, []

    t0 = t_burst
    t1 = min(t_burst + T_EFFECT, T_HIT_loc)
    if t1 <= t0:
        return 0.0, []

    intervals: List[Tuple[float, float]] = []
    in_seg = False
    seg_start = None

    t = t0
    while t <= t1 + 1e-12:
        flag = covered_L1_at_time_vectorized(M0, s_burst, t_burst, t, PTS)
        if flag and not in_seg:
            in_seg, seg_start = True, t
        if (not flag) and in_seg:
            in_seg = False
            intervals.append((seg_start, t))
        t += dt
    if in_seg:
        intervals.append((seg_start, t1))

    intervals = _canonize_intervals(intervals, t0, t1, dt)
    covered = sum(b - a for (a, b) in intervals)
    return covered, intervals

# =========================
# 六、PSO（通用）
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
        for _ in range(self.iters):
            w = self.w
            for i in range(self.swarm_size):
                r1 = self.rng.random(len(self.bounds))
                r2 = self.rng.random(len(self.bounds))
                V[i] = w * V[i] + self.c1 * r1 * (pbest_X[i] - X[i]) + self.c2 * r2 * (gbest_x - X[i])
                X[i] = X[i] + V[i]
                # 边界/角度处理
                for j, (lo, hi) in enumerate(self.bounds):
                    if j == 0:  # theta
                        X[i, j] = X[i, j] % (2.0 * math.pi)
                    else:
                        if X[i, j] < lo or X[i, j] > hi:
                            X[i, j] = clip_bounds(X[i, j], lo, hi)
                            V[i, j] *= -0.5
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
# 七、统一求解接口（L0 / L1 / two_stage）
# =========================

def solve_q2(strategy: str = "two_stage",
             # L0 / L1 数值参数
             dt_L0: float = 0.01,
             N_ANG: int = 48, N_Z: int = 9, INCLUDE_SIDE: bool = True, dt_L1: float = 0.02,
             # PSO 参数（阶段1）
             swarm_size: int = 64, iters: int = 120, seed: int = 2025,
             # 两阶段阶段2（L1微调）PSO 参数
             stage2_swarm: int = 48, stage2_iters: int = 80,
             # 可选初始化提示：(theta, v, t_drop, tau)
             init_hint: Optional[Tuple[float, float, float, float]] = None
             ) -> Dict[str, Any]:
    """
        统一入口：根据 strategy 选择 L0 / L1 / two_stage。
    """
    bounds = [
        (0.0, 2.0 * math.pi),  # theta
        (70.0, 140.0),         # v_u
        (0.0, 60.0),           # t_drop
        (0.2, 12.0),           # tau
    ]

    # 预构建 L1 采样点（仅在需要时）
    PTS = None
    if strategy in ("L1", "two_stage"):
        PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)

    # 定义 f_eval
    def f_eval_L0(theta, v, t_drop, tau):
        return eval_cover_time_L0(theta, v, t_drop, tau, dt=dt_L0)

    def f_eval_L1(theta, v, t_drop, tau):
        return eval_cover_time_L1(theta, v, t_drop, tau, PTS=PTS, dt=dt_L1)

    # 处理 init_hint
    init_vec = None
    if init_hint is not None:
        th, vv, td, ta = init_hint
        init_vec = np.array([
            float(th) % (2.0 * math.pi),
            clip_bounds(float(vv), bounds[1][0], bounds[1][1]),
            clip_bounds(float(td), bounds[2][0], bounds[2][1]),
            clip_bounds(float(ta), bounds[3][0], bounds[3][1]),
        ], dtype=float)

    # 求解
    if strategy == "L0":
        pso = PSO(f_eval_L0, bounds, swarm_size=swarm_size, iters=iters, seed=seed, init_hint=init_vec)
        best_x, best_cover, info = pso.optimize()
        eval_used = "L0"

    elif strategy == "L1":
        pso = PSO(f_eval_L1, bounds, swarm_size=swarm_size, iters=iters, seed=seed, init_hint=init_vec)
        best_x, best_cover, info = pso.optimize()
        eval_used = "L1"

    elif strategy == "two_stage":
        # 阶段1：L0 全局
        pso1 = PSO(f_eval_L0, bounds, swarm_size=swarm_size, iters=iters, seed=seed, init_hint=init_vec)
        x1, cover1, info1 = pso1.optimize()
        # 阶段2：L1 小规模精修
        pso2 = PSO(f_eval_L1, bounds, swarm_size=stage2_swarm, iters=stage2_iters, seed=seed+1, init_hint=x1)
        best_x, best_cover, info = pso2.optimize()
        eval_used = "two_stage"
    else:
        raise ValueError("strategy 必须为 'L0'、'L1' 或 'two_stage'")

    # 整理输出
    theta = info["theta"]
    v = info["v"]
    t_drop = info["t_drop"]
    tau = info["tau"]
    t_burst = t_drop + tau
    s_burst = burst_point(U0, theta, v, t_drop, tau)

    precise_intervals = info["intervals"]
    pretty_intervals  = [(round(a, 3), round(b, 3)) for (a, b) in precise_intervals]

    out = {
        "strategy": eval_used,
        "theta_deg": (math.degrees(theta) % 360.0),  # 0~360, 逆时针为正
        "v_u_mps": v,
        "t_drop_s": t_drop,
        "tau_s": tau,
        "t_burst_s": t_burst,
        "burst_point_m": (float(s_burst[0]), float(s_burst[1]), float(s_burst[2])),
        "cover_total_s": best_cover,
        "cover_intervals_s": precise_intervals,      # 精确版（校验用）
        "cover_intervals_pretty": pretty_intervals,  # 仅展示
        "config": {
            "dt_L0": dt_L0, "dt_L1": dt_L1,
            "N_ANG": N_ANG, "N_Z": N_Z, "INCLUDE_SIDE": INCLUDE_SIDE,
            "swarm_size": swarm_size, "iters": iters,
            "stage2_swarm": stage2_swarm, "stage2_iters": stage2_iters
        }
    }
    return out

# =========================
# 八、模型检验与工具（原样，略有兼容）
# =========================

def _intervals_ok(intervals, t0, t1, eps=1e-6):
    """区间有序、互不重叠、在 [t0,t1] 内"""
    if not intervals: return True
    last_b = None
    for a,b in intervals:
        if not (t0 - eps <= a <= t1 + eps and t0 - eps <= b <= t1 + eps and a <= b + eps):
            return False
        if last_b is not None and a < last_b - eps:
            return False
        last_b = b
    return True

def _coverage_upper_bound_ok(cover, t_burst, T_HIT, eps=1e-6):
    """覆盖时长不超过物理上界：min(T_EFFECT, T_HIT - t_burst)"""
    ub = max(0.0, min(T_EFFECT, T_HIT - t_burst))
    return cover <= ub + 2e-3  # 少量离散化富余

def _solve_and_unpack(strategy, **kw):
    out = solve_q2(strategy=strategy, **kw)
    th = float(out["theta_deg"]) * math.pi/180.0
    v  = float(out["v_u_mps"])
    td = float(out["t_drop_s"])
    ta = float(out["tau_s"])
    tb = td + ta
    cov= float(out["cover_total_s"])
    intervals = out["cover_intervals_s"]
    return out, (th, v, td, ta, tb, cov, intervals)

def validate_intervals_consistency(ans, dt_ref=0.01):
    """区间与上界一致性 + 基本物理边界"""
    th = ans["theta_deg"] * math.pi/180.0
    v  = ans["v_u_mps"]; td = ans["t_drop_s"]; ta = ans["tau_s"]
    tb = td + ta; cov = ans["cover_total_s"]; intervals = ans["cover_intervals_s"]
    assert 70.0 - 1e-6 <= v <= 140.0 + 1e-6, "v_u 越界"
    assert 0.0 - 1e-6 <= td <= 60.0 + 1e-6,  "t_drop 越界"
    assert 0.2 - 1e-6 <= ta <= 12.0 + 1e-6,  "tau 越界"
    # 区间放在有效窗 [tb, min(tb+T_EFFECT, T_HIT)]
    T_HIT_loc = missile_hit_time(M0)
    t0 = tb
    t1 = min(tb + T_EFFECT, T_HIT_loc)
    assert _intervals_ok(intervals, t0, t1), "遮蔽区间非法（顺序/边界/重叠）"
    assert _coverage_upper_bound_ok(cov, tb, T_HIT_loc), "覆盖超过理论上界"

def quick_grid_upper_bound(strategy="L0",
                           th_list=None, v_list=None, td_list=None, ta_list=None,
                           dt_L0=0.02, dt_L1=0.03,
                           N_ANG=32, N_Z=7, INCLUDE_SIDE=True):
    """粗网格把关：返回网格最优覆盖，作为 sanity baseline"""
    if th_list is None:
        th_list = [0, 45, 90, 135, 180, 225, 270, 315]  # 度
    th_list = [math.radians(x) for x in th_list]
    if v_list is None:
        v_list = [70, 90, 110, 130, 140]
    if td_list is None:
        td_list = [0, 5, 10, 15, 20, 30, 40, 50, 60]
    if ta_list is None:
        ta_list = [0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 9.0, 12.0]

    if strategy == "L1":
        PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)
        def f(th, v, td, ta): return eval_cover_time_L1(th, v, td, ta, PTS=PTS, dt=dt_L1)[0]
    else:
        def f(th, v, td, ta): return eval_cover_time_L0(th, v, td, ta, dt=dt_L0)[0]

    best = 0.0
    for th in th_list:
        for v in v_list:
            for td in td_list:
                for ta in ta_list:
                    best = max(best, f(th, v, td, ta))
    return best

def convergence_test_dt(base_strategy="L0",
                        dts=(0.04, 0.02, 0.01, 0.005),
                        N_ANG=48, N_Z=9, INCLUDE_SIDE=True):
    """时间步长收敛性（覆盖时长是否趋于稳定）"""
    rows=[]
    init_hint=None
    for k,dt in enumerate(dts):
        if base_strategy=="L1":
            out = solve_q2(strategy="L1", dt_L1=dt, N_ANG=N_ANG, N_Z=N_Z, INCLUDE_SIDE=INCLUDE_SIDE,
                           swarm_size=48, iters=90, init_hint=init_hint)
        else:
            out = solve_q2(strategy="L0", dt_L0=dt, swarm_size=48, iters=90, init_hint=init_hint)
        cov = out["cover_total_s"]; rows.append((dt, cov))
        init_hint = (math.radians(out["theta_deg"]), out["v_u_mps"], out["t_drop_s"], out["tau_s"])
    return rows  # [(dt, cover), ...]

def rotation_invariance_test(angles_deg=(0, 30, 90, 150),
                             strategy="two_stage",
                             dt_L0=0.01, dt_L1=0.02,
                             N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
                             tol=0.5):
    """
    旋转不变性：把 (M0, U0, P_TARGET, CYL_CENTER) 同时绕 z 轴旋转若干角度，
    期望覆盖时长近似不变（容差 tol 秒）
    """
    def rotz(vec, ang_rad):
        c,s = math.cos(ang_rad), math.sin(ang_rad)
        x,y,z = vec[0], vec[1], vec[2]
        return np.array([c*x - s*y, s*x + c*y, z], dtype=float)

    # 备份
    M0_bak = M0.copy()
    U0_bak = U0.copy()
    P_bak  = P_TARGET.copy()
    C_bak  = CYL_CENTER.copy()

    covs=[]
    for ang in angles_deg:
        rad = math.radians(ang)
        # 改写全局（旋转整场景）
        M0[:] = rotz(M0_bak, rad)
        U0[:] = rotz(U0_bak, rad)
        P_TARGET[:] = rotz(P_bak, rad)
        CYL_CENTER[:] = rotz(C_bak, rad)
        out = solve_q2(strategy=strategy,
                       dt_L0=dt_L0, dt_L1=dt_L1,
                       N_ANG=N_ANG, N_Z=N_Z, INCLUDE_SIDE=INCLUDE_SIDE,
                       swarm_size=48, iters=90, stage2_swarm=36, stage2_iters=60)
        covs.append(out["cover_total_s"])

    # 复原
    M0[:] = M0_bak; U0[:] = U0_bak; P_TARGET[:] = P_bak; CYL_CENTER[:] = C_bak

    # 检查波动
    if covs:
        if max(covs) - min(covs) > tol:
            print(f"[warn] 旋转不变性偏差较大：{covs}")
    return list(zip(angles_deg, covs))

def sensitivity_test(strategy="two_stage",
                     noise_pos=5.0,   # 位置扰动半径（米）
                     noise_param=0.03,# 比例扰动（R_SMOKE、V_SINK）
                     trials=10,
                     dt_L0=0.01, dt_L1=0.02,
                     N_ANG=48, N_Z=9, INCLUDE_SIDE=True, seed=42):
    """
    对 M0/U0 坐标、R_SMOKE/V_SINK 做小扰动，观察覆盖分布
    """
    rng = np.random.default_rng(seed)
    covs=[]
    # 备份
    M0_bak = M0.copy(); U0_bak = U0.copy()
    P_bak  = P_TARGET.copy(); C_bak = CYL_CENTER.copy()
    global R_SMOKE, V_SINK
    R_bak, S_bak = R_SMOKE, V_SINK
    for _ in range(trials):
        dxy1 = rng.normal(0.0, noise_pos, size=2)
        dxy2 = rng.normal(0.0, noise_pos, size=2)
        M0[:2] = M0_bak[:2] + dxy1
        U0[:2] = U0_bak[:2] + dxy2
        R_SMOKE = max(5.0, R_bak * (1.0 + rng.normal(0.0, noise_param)))
        V_SINK  = max(1.0, S_bak * (1.0 + rng.normal(0.0, noise_param)))
        out = solve_q2(strategy=strategy,
                       dt_L0=dt_L0, dt_L1=dt_L1,
                       N_ANG=N_ANG, N_Z=N_Z, INCLUDE_SIDE=INCLUDE_SIDE,
                       swarm_size=48, iters=90, stage2_swarm=36, stage2_iters=60)
        covs.append(out["cover_total_s"])
    # 复原
    M0[:] = M0_bak; U0[:] = U0_bak; P_TARGET[:] = P_bak; CYL_CENTER[:] = C_bak
    R_SMOKE, V_SINK = R_bak, S_bak
    covs = np.array(covs, dtype=float)
    return {"mean": float(covs.mean()), "std": float(covs.std()), "min": float(covs.min()), "max": float(covs.max())}

def multi_seed_stability(strategy="two_stage",
                         seeds=(2025, 7, 17, 99),
                         dt_L0=0.01, dt_L1=0.02,
                         N_ANG=48, N_Z=9, INCLUDE_SIDE=True):
    """不同随机种子下的覆盖波动"""
    covs=[]
    for sd in seeds:
        out = solve_q2(strategy=strategy,
                       dt_L0=dt_L0, dt_L1=dt_L1,
                       N_ANG=N_ANG, N_Z=N_Z, INCLUDE_SIDE=INCLUDE_SIDE,
                       swarm_size=64, iters=120, stage2_swarm=48, stage2_iters=80,
                       init_hint=None)
        covs.append(out["cover_total_s"])
    covs = np.array(covs, dtype=float)
    return {"mean": float(covs.mean()), "std": float(covs.std()), "values": list(map(float, covs))}

def run_all_validations():
    print("\n[check] 1) 三策略求解与基本一致性")
    out_L0,_ = _solve_and_unpack("L0", dt_L0=0.01, swarm_size=64, iters=120)
    out_L1,_ = _solve_and_unpack("L1", dt_L1=0.02, N_ANG=48, N_Z=9, INCLUDE_SIDE=True, swarm_size=64, iters=120)
    out_2s,_ = _solve_and_unpack("two_stage",
                                 dt_L0=0.01, dt_L1=0.02,
                                 N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
                                 swarm_size=64, iters=120,
                                 stage2_swarm=48, stage2_iters=80)
    for tag,ans in [("L0", out_L0), ("L1", out_L1), ("two_stage", out_2s)]:
        validate_intervals_consistency(ans)
        print(f"  - {tag}: cover={ans['cover_total_s']:.3f}s, intervals={ans['cover_intervals_s']}")

    print("\n[check] 2) 粗网格 baseline 与 PSO 解对比（L0/L1）")
    coarse_L0 = quick_grid_upper_bound("L0", dt_L0=0.03)
    coarse_L1 = quick_grid_upper_bound("L1", dt_L1=0.04, N_ANG=32, N_Z=7)
    print(f"  - coarse_L0≈{coarse_L0:.3f}s, PSO_L0={out_L0['cover_total_s']:.3f}s")
    print(f"  - coarse_L1≈{coarse_L1:.3f}s, PSO_L1={out_L1['cover_total_s']:.3f}s")
    assert out_L0["cover_total_s"] >= coarse_L0 - 0.6, "PSO(L0) 未达粗网格基线"
    assert out_L1["cover_total_s"] >= coarse_L1 - 0.6, "PSO(L1) 未达粗网格基线"

    print("\n[check] 3) 时间步长收敛（L0）")
    rows_L0 = convergence_test_dt("L0", dts=(0.04,0.02,0.01,0.005))
    print("  dt vs cover:", rows_L0)
    assert abs(rows_L0[-1][1]-rows_L0[-2][1]) <= 0.4, "L0 覆盖随 dt 未收敛到 0.4s 内"

    print("\n[check] 4) 旋转不变性（two_stage）")
    rotrows = rotation_invariance_test(angles_deg=(0, 45, 90, 135), strategy="two_stage")
    print("  angle vs cover:", rotrows)

    print("\n[check] 5) 小扰动敏感性（two_stage）")
    sens = sensitivity_test(strategy="two_stage", trials=8)
    print("  sensitivity:", sens)

    print("\n[check] 6) 多 seed 稳定性（two_stage）")
    st = multi_seed_stability(strategy="two_stage", seeds=(2025,7,17,99))
    print("  stability:", st)

# ========= 可视化：时间条 + 3D球 =========
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def _pick_focus_time(t_burst, intervals, dt=0.01):
    """优先选‘最长遮蔽段’的中点；若无遮蔽，则选与视轴最近时刻"""
    if intervals:
        a, b = max(intervals, key=lambda ab: ab[1] - ab[0])
        return 0.5 * (a + b)
    # 无遮蔽：取最近距离时刻
    T_HIT_loc = missile_hit_time(M0)
    t0, t1 = t_burst, min(t_burst + T_EFFECT, T_HIT_loc)
    ts = np.arange(t0, t1 + 1e-12, dt)
    best_t, best_d = t0, 1e18
    for tt in ts:
        d = point_to_segment_dist(P_TARGET, missile_pos(M0, tt), smoke_center_after_burst(
            burst_point(U0, math.radians(0), 0, 0, 0), tt, 0))  # 仅占位，不用这个值
    # 实际用球心直接算
    s_burst_dummy = np.zeros(3)  # 只是为了兼容写法，不用它
    for tt in ts:
        m = missile_pos(M0, tt)
        # 最近距离 = 点到直线段(P_TARGET, m)的距离；球心此处只用视角关系，无需真实值
        d = point_to_segment_dist(P_TARGET, m, P_TARGET)  # 与视轴的“本征尺度”比较
        if d < best_d:
            best_d, best_t = d, tt
    return best_t

def plot_cover_timeline(ans, ax=None):
    """遮蔽区间时间条"""
    tb = float(ans["t_burst_s"])
    intervals = ans["cover_intervals_s"]
    T_HIT_loc = missile_hit_time(M0)
    t0, t1 = tb, min(tb + T_EFFECT, T_HIT_loc)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.8, 1.8), constrained_layout=True)
    ax.hlines(1, t0, t1, color="#dddddd", lw=8, label="有效窗口")
    for a, b in intervals:
        ax.hlines(1, a, b, color="tab:red", lw=8, label="遮蔽段")
    ax.set_ylim(0.8, 1.2); ax.set_yticks([])
    ax.set_xlabel("t / s"); ax.set_title("遮蔽时间区间")
    # 去重 legend
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right")
    return ax

def plot_cover_3d(ans, save_path=None):
    """完整球 + 视轴（球内实/球外虚）+ 球心与垂足 P（第二张图风格）"""
    th = math.radians(float(ans["theta_deg"]))
    v  = float(ans["v_u_mps"])
    td = float(ans["t_drop_s"])
    ta = float(ans["tau_s"])
    tb = td + ta

    # 起爆点、焦点时刻
    s_burst = burst_point(U0, th, v, td, ta)
    t_focus = _pick_focus_time(tb, ans["cover_intervals_s"], dt=0.01)

    # 焦点的导弹位置、球心位置、视轴方向
    m = missile_pos(M0, t_focus)
    s = smoke_center_after_burst(s_burst, t_focus, tb)
    u = unit(m - P_TARGET)  # 视轴方向（圆柱中心 -> 导弹）

    # 与球的交点（用直线：P_TARGET + u * t）
    w = P_TARGET - s
    b = 2.0 * float(np.dot(u, w))
    c = float(np.dot(w, w)) - R_SMOKE**2
    disc = b*b - 4.0*c
    have_intersection = disc >= 0.0
    if have_intersection:
        sqrtD = math.sqrt(disc)
        t1 = (-b - sqrtD) / 2.0
        t2 = (-b + sqrtD) / 2.0
        if t1 > t2: t1, t2 = t2, t1
        P_in  = P_TARGET + u * t1
        P_out = P_TARGET + u * t2

    # 球心到视轴的垂足
    # 视轴的参考点取 P_TARGET，方向 u
    t_foot = float(np.dot(s - P_TARGET, u))
    P_foot = P_TARGET + u * t_foot

    # --- 画图 ---
    fig = plt.figure(figsize=(8.6, 6.6), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title("遮掩示意图")

    # 球体（完整）
    ugrid = np.linspace(0, 2*np.pi, 80)
    vgrid = np.linspace(0,   np.pi, 40)
    xs = s[0] + R_SMOKE*np.outer(np.cos(ugrid), np.sin(vgrid))
    ys = s[1] + R_SMOKE*np.outer(np.sin(ugrid), np.cos(vgrid))  # 注意：这里改成 cos(v) 会变椭球观感
    ys = s[1] + R_SMOKE*np.outer(np.sin(ugrid), np.sin(vgrid))
    zs = s[2] + R_SMOKE*np.outer(np.ones_like(ugrid), np.cos(vgrid))
    ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, linewidth=0.3,
                    alpha=0.25, color='tab:red', edgecolor='k')

    # 视轴：球内实线、球外虚线
    L = 2.2 * R_SMOKE
    if have_intersection:
        ax.plot([P_in[0], P_out[0]], [P_in[1], P_out[1]], [P_in[2], P_out[2]],
                'k-', lw=2.2, label='导弹视轴（球内实线）')
        L1 = P_in - u * L
        L2 = P_out + u * L
        ax.plot([L1[0], P_in[0]], [L1[1], P_in[1]], [L1[2], P_in[2]], 'k--', lw=2.2)
        ax.plot([P_out[0], L2[0]], [P_out[1], L2[1]], [P_out[2], L2[2]], 'k--', lw=2.2)
    else:
        O1 = P_TARGET - u * L; O2 = P_TARGET + u * L
        ax.plot([O1[0], O2[0]], [O1[1], O2[1]], [O1[2], O2[2]], 'k--', lw=2.2, label='导弹视轴')

    # 球心与垂足 P + 垂线
    ax.scatter([s[0]], [s[1]], [s[2]], c='tab:red', s=28, label='球心')
    ax.scatter([P_foot[0]], [P_foot[1]], [P_foot[2]], c='k', s=28, label='垂足 P')
    ax.plot([s[0], P_foot[0]], [s[1], P_foot[1]], [s[2], P_foot[2]], color='tab:red', lw=2.0)

    # 等比例 & 紧凑取景：以球心为中心的立方体
    pts = [s, P_foot]
    if have_intersection: pts += [P_in, P_out]
    pts = np.array(pts)
    max_dev = float(np.max(np.abs(pts - s)))
    half = max(1.35 * R_SMOKE, max_dev + 0.35 * R_SMOKE)
    ax.set_xlim(s[0]-half, s[0]+half)
    ax.set_ylim(s[1]-half, s[1]+half)
    ax.set_zlim(s[2]-half, s[2]+half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('X / m'); ax.set_ylabel('Y / m'); ax.set_zlabel('Z / m')
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc='upper right', fontsize=9)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=180)
        print(f"[OK] 图已保存：{save_path}")
    plt.show()

def _rand_points_on_cyl_surface(n, rng, R=R_TAR, H=H_TAR, center=CYL_CENTER):
    """按面积权重随机采样：侧壁+上下底面"""
    cx, cy, z0 = center
    A_side = 2.0 * math.pi * R * H
    A_cap  = math.pi * R * R
    probs = np.array([A_side, A_cap, A_cap], dtype=float)
    probs /= probs.sum()

    pts = []
    for _ in range(n):
        face = rng.choice(3, p=probs)
        if face == 0:
            # 侧壁
            ang = rng.uniform(0, 2*math.pi)
            z   = rng.uniform(0, H)
            x = cx + R * math.cos(ang)
            y = cy + R * math.sin(ang)
            pts.append((x, y, z))
        elif face == 1:
            # 底面 z=0
            ang = rng.uniform(0, 2*math.pi)
            r   = R * math.sqrt(rng.uniform(0, 1))
            x = cx + r * math.cos(ang)
            y = cy + r * math.sin(ang)
            pts.append((x, y, 0.0))
        else:
            # 顶面 z=H
            ang = rng.uniform(0, 2*math.pi)
            r   = R * math.sqrt(rng.uniform(0, 1))
            x = cx + r * math.cos(ang)
            y = cy + r * math.sin(ang)
            pts.append((x, y, H))
    return np.array(pts, dtype=float)

def _line_sphere_intersection(P, uhat, S, R):
    """线 (P + t*uhat, t∈R, |uhat|=1) 与球 |X-S|=R 相交求解；返回 (hit, t1, t2, P_perp, r_perp)"""
    w = P - S
    b = 2.0 * float(np.dot(uhat, w))
    c = float(np.dot(w, w)) - R*R
    disc = b*b - 4.0*c
    # 垂足与垂距
    t_perp = -0.5 * b        # 因为 a=1
    P_perp = P + t_perp * uhat
    r_perp = float(np.linalg.norm(S - P_perp))
    if disc < 0.0:
        return False, None, None, P_perp, r_perp
    sqrtD = math.sqrt(disc)
    t1 = (-b - sqrtD) / 2.0
    t2 = (-b + sqrtD) / 2.0
    if t1 > t2: t1, t2 = t2, t1
    return True, t1, t2, P_perp, r_perp

def _mid_of_longest_interval(intervals, tb, dt=0.01):
    if intervals:
        a, b = max(intervals, key=lambda ab: ab[1]-ab[0])
        return 0.5*(a+b)
    # 退化：没有遮蔽区间，取有效窗中点
    T_HIT_loc = missile_hit_time(M0)
    t0, t1 = tb, min(tb + T_EFFECT, T_HIT_loc)
    return 0.5*(t0+t1)

def plot_cover_3d_random_rays(ans,
                              n_points=10,
                              seed=2026,
                              require_exact_one=True,
                              max_trials=500,
                              outside_len=0.8,   # 虚线长度系数（×R）
                              overshoot=0.20,    # 实线入射端向外“掏”出一点（×R）
                              save_path="result/Problem2_result/Q2_multi10.png"):
    """
    像单条示意那样画球，但将视轴改为：在圆柱表面随机取 n_points 个点→每点连到导弹，
    只在烟雾附近显示；强制使“恰好一条穿过烟雾”，其余都不穿过（若可行）。
    """
    th = math.radians(float(ans["theta_deg"]))
    v  = float(ans["v_u_mps"])
    td = float(ans["t_drop_s"])
    ta = float(ans["tau_s"])
    tb = td + ta

    # 选一个“发生遮蔽”的时刻（取 L1/2s 输出的最长区间中点；没有则用有效窗中点）
    t_focus = _mid_of_longest_interval(ans["cover_intervals_s"], tb, dt=0.01)

    # 当时刻的几何
    s_burst = burst_point(U0, th, v, td, ta)
    S = smoke_center_after_burst(s_burst, t_focus, tb)
    M = missile_pos(M0, t_focus)

    rng = np.random.default_rng(seed)

    # 反复抽样，尽量满足“恰好一条命中”
    chosen = None
    info   = None
    for _ in range(max_trials):
        PTS = _rand_points_on_cyl_surface(n_points, rng)
        hits = []
        segs = []
        for P in PTS:
            uhat = unit(M - P)
            hit, t1, t2, P_perp, r_perp = _line_sphere_intersection(P, uhat, S, R_SMOKE)
            segs.append((P, uhat, hit, t1, t2, P_perp, r_perp))
            if hit and t2 >= 0.0:   # 往导弹方向的相交
                hits.append(True)
            else:
                hits.append(False)
        k = sum(hits)
        if (require_exact_one and k == 1) or (not require_exact_one):
            chosen, info = PTS, segs
            break
    # 如果实在凑不出恰好一条，就退一步：选命中数量最接近 1 的那次
    if chosen is None:
        chosen = PTS
        info   = segs

    # ---- 画图 ----
    plt.rcParams['font.family'] = 'SimHei'; plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(8.8, 6.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("遮掩示意图（圆柱面随机10点；恰好1条穿球）", pad=14)

    # 球
    u = np.linspace(0, 2*np.pi, 120); v = np.linspace(0, np.pi, 60)
    xs = S[0] + R_SMOKE*np.outer(np.cos(u), np.sin(v))
    ys = S[1] + R_SMOKE*np.outer(np.sin(u), np.sin(v))
    zs = S[2] + R_SMOKE*np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='tab:red', alpha=0.28,
                    edgecolor='k', linewidth=0.25)
    ax.scatter([S[0]], [S[1]], [S[2]], c='tab:red', s=28, label='球心')

    # 画每条视轴：球内实线，球外短虚线，并标垂足
    Lout = outside_len * R_SMOKE
    n_hit = 0
    all_pts = [S.copy()]
    for (P, uhat, hit, t1, t2, P_perp, r_perp) in info:
        if hit and t2 >= 0.0:
            n_hit += 1
            # 只画球内段（入口→出口），入口再向外“掏”一点增强可见
            Pin  = P + t1 * uhat
            Pout = P + t2 * uhat
            Pin2 = Pin - overshoot * (np.linalg.norm(Pout-Pin)) * uhat
            ax.plot([Pin2[0], Pout[0]], [Pin2[1], Pout[1]], [Pin2[2], Pout[2]],
                    'k-', lw=2.2, label='导弹视轴（球内实线）' if n_hit==1 else None)
            # 两侧短虚线
            ax.plot([Pin2[0]-Lout*uhat[0], Pin2[0]], [Pin2[1]-Lout*uhat[1], Pin2[1]],
                    [Pin2[2]-Lout*uhat[2], Pin2[2]], 'k--', lw=2.0)
            ax.plot([Pout[0], Pout[0]+Lout*uhat[0]], [Pout[1], Pout[1]+Lout*uhat[1]],
                    [Pout[2], Pout[2]+Lout*uhat[2]], 'k--', lw=2.0)
            all_pts += [Pin2, Pout]
        else:
            # miss：以垂足为中心画一段短虚线
            a = Lout
            A = P_perp - a*uhat; B = P_perp + a*uhat
            ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], '--', color='0.5', lw=1.8,
                    label=None)
        # 垂足点
        ax.scatter([P_perp[0]], [P_perp[1]], [P_perp[2]], c='k', s=24, label='垂足 P')

    # 紧凑取景（围绕球）
    all_pts = np.array(all_pts)
    max_dev = float(np.max(np.abs(all_pts - S)))
    half = max(1.35*R_SMOKE, max_dev + 0.35*R_SMOKE)
    ax.set_xlim(S[0]-half, S[0]+half)
    ax.set_ylim(S[1]-half, S[1]+half)
    ax.set_zlim(S[2]-half, S[2]+half)
    ax.set_box_aspect((1,1,1))
    ax.set_xlabel("X / m"); ax.set_ylabel("Y / m"); ax.set_zlabel("Z / m")
    ax.view_init(elev=22, azim=-55)
    # 去重 legend
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc='upper right', fontsize=9)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[OK] 已保存：{save_path}")
    plt.show()

# =========================
# 九、示例调用
# =========================
if __name__ == "__main__":
    # # 1) L0：最快的近似版
    # ans_L0 = solve_q2(strategy="L0", dt_L0=0.01, swarm_size=64, iters=120)
    # print("[Q2 | L0] 最优解：")
    # for k, v in ans_L0.items():
    #     print("  ", k, ":", v)
    #
    # # 2) L1：高保真直接求（可略加大群体/迭代）
    ans_L1 = solve_q2(strategy="L1", N_ANG=48, N_Z=9, INCLUDE_SIDE=True, dt_L1=0.02,
                      swarm_size=64, iters=120)
    print("\n[Q2 | L1] 最优解：")
    for k, v in ans_L1.items():
        print("  ", k, ":", v)
    #
    # # 3) two_stage：先 L0 全局搜，再 L1 小规模精修（推荐）
    ans_2s = solve_q2(strategy="two_stage",
                      dt_L0=0.01, swarm_size=64, iters=120,
                      N_ANG=48, N_Z=9, INCLUDE_SIDE=True, dt_L1=0.02,
                      stage2_swarm=48, stage2_iters=80)
    print("\n[Q2 | two_stage] 最优解：")
    for k, v in ans_2s.items():
        print("  ", k, ":", v)
    # plot_cover_3d_random_rays(ans_2s, n_points=10, seed=2026,
    #                           require_exact_one=True,
    #                           save_path="result/Problem2_result/Q2_multi10.png")
    # 时间条
    plot_cover_timeline(ans_2s)

    # 3D 示意（保存可选）
    plot_cover_3d(ans_2s, save_path="result/Problem2_result/Q2.png")

    #run_all_validations()
