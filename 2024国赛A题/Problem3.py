import math
import numpy as np

# ====== 你已有的参数（保留） ======
R_TURN = 4.5                      # 调头圆半径（m），题面直径9m的一半
THETA0 = 2.0 * math.pi * 16       # 初始在第16圈A点（θ0=16*2π）
V_HEAD = 1.0                      # 龙头弧长速度恒定 = 1 m/s
W_BOARD = 0.30                    # 板宽（m）
EXT_END = 0.275                   # 孔心到两端头延伸（m）
l_HEAD, l_BODY, l_TAIL = 2.86, 1.65, 1.65   # 同板前后把手中心距（m），与你保持一致

# ====== 你已有的弧长原函数 F(θ) 与反函数 inv_F (保持不变) ======
def F_theta(theta: float) -> float:
    """F(θ) = 0.5 * (θ*sqrt(1+θ^2) + asinh(θ))"""
    return 0.5 * (theta * math.sqrt(1.0 + theta * theta) + math.asinh(theta))

def inv_F(Fv: float, max_iter: int = 40, tol: float = 1e-13) -> float:
    """牛顿法反解 θ，使 F(θ) = Fv（θ≥0）"""
    theta = math.sqrt(2.0 * Fv) if Fv > 1.0 else Fv
    for _ in range(max_iter):
        f = F_theta(theta) - Fv
        df = math.sqrt(1.0 + theta * theta)
        theta_new = theta - f / df
        if theta_new < 0.0:
            theta_new = 0.0
        if abs(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new
    return theta

# ====== 按给定 b=p/(2π) 计算龙头在时刻 t 的极坐标 ======
def head_polar_at_time_b(t: float, b: float) -> tuple[float, float]:
    """
    给定时间 t 与参数 b，按 v=1 的解析关系：
      F(θ(t)) = F(θ0) - t/b
      r(t) = b*θ(t)
    """
    F0 = F_theta(THETA0)
    Fv = F0 - t / b
    if Fv <= 0.0:
        # θ -> 0 的极限（已经到中心附近）
        theta = 0.0
    else:
        theta = inv_F(Fv)
    r = b * theta
    return r, theta

# ====== 你已有的“同板前→后”角度增量求解，推广到传入 b ======
def solve_delta_same_board_b(theta1: float, l: float, b: float) -> float:
    """
    已知同螺线上的前把手角 θ1，求同板后把手角 θ2=θ1+Δ，使两点直线距为 l。
    方程：g(Δ)=r1^2 + r2^2 - 2 r1 r2 cosΔ - l^2 = 0
    """
    r1 = b * theta1
    def g(delta: float) -> float:
        r2 = b * (theta1 + delta)
        return r1*r1 + r2*r2 - 2.0*r1*r2*math.cos(delta) - l*l
    def gprime(delta: float) -> float:
        r2 = b * (theta1 + delta)
        return 2.0*r2*b - 2.0*r1*b*math.cos(delta) + 2.0*r1*r2*math.sin(delta)

    # 初值：小角近似 Δ≈l/max(r1, ε)
    delta = min(max(l / max(r1, 1e-9), 1e-10), math.pi/2)
    ok = False
    for _ in range(20):
        val = g(delta)
        der = gprime(delta)
        if abs(der) < 1e-14:
            break
        cand = delta - val/der
        if not (0.0 < cand < math.pi):
            cand = 0.5*(delta + max(1e-10, min(cand, math.pi-1e-10)))
        delta = cand
        if abs(val) < 1e-12:
            ok = True
            break
    if ok: return delta

    # 退化则二分
    lo, hi = 1e-12, math.pi - 1e-12
    g_lo, g_hi = g(lo), g(hi)
    if g_lo * g_hi > 0:
        return delta
    for _ in range(80):
        mid = 0.5*(lo + hi)
        gm = g(mid)
        if gm == 0.0 or (hi - lo) < 1e-12:
            return mid
        if g_lo * gm <= 0.0:
            hi, g_hi = mid, gm
        else:
            lo, g_lo = mid, gm
    return 0.5*(lo + hi)

def step_same_board_b(theta1: float, l: float, b: float) -> tuple[float, float]:
    """同板前→后（推广到传入 b），返回 (r2, θ2)"""
    delta = solve_delta_same_board_b(theta1, l, b)
    theta2 = theta1 + delta
    return b * theta2, theta2

# ====== 生成 t 时刻所有把手（推广到传入 b） ======
def handles_at_time_b(t: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """
    返回 (X, Y)，只保留坐标，顺序与你的 handles_at_time 保持一致：
      龙头前 -> 第1..221节龙身前 -> 龙尾前 -> 龙尾后
    """
    # 龙头前把手
    r, th = head_polar_at_time_b(t, b)
    xs = [r * math.cos(th)]
    ys = [r * math.sin(th)]

    # 第1节龙身前把手 = “龙头同板后把手”
    r_f, th_f = step_same_board_b(th, l_HEAD, b)
    xs.append(r_f * math.cos(th_f)); ys.append(r_f * math.sin(th_f))

    # 2..221节
    th_cur = th_f
    for _ in range(2, 222):
        r_f, th_f = step_same_board_b(th_cur, l_BODY, b)
        xs.append(r_f * math.cos(th_f)); ys.append(r_f * math.sin(th_f))
        th_cur = th_f

    # 龙尾前把手（= 第221节后把手）
    r_back_221, th_back_221 = step_same_board_b(th_cur, l_BODY, b)
    xs.append(r_back_221 * math.cos(th_back_221)); ys.append(r_back_221 * math.sin(th_back_221))

    # 龙尾后把手
    r_tail, th_tail = step_same_board_b(th_back_221, l_TAIL, b)
    xs.append(r_tail * math.cos(th_tail)); ys.append(r_tail * math.sin(th_tail))

    return np.array(xs), np.array(ys)

# ====== 由把手生成 223 块矩形板（平端），并做宽相剔除 + SAT 判碰 ======
def _norm(v: np.ndarray, eps: float = 1e-12) -> float:
    n = float(np.hypot(v[0], v[1])); return n if n > eps else eps

def _unit(v: np.ndarray) -> np.ndarray:
    n = _norm(v); return v / n

def build_board_segments_from_handles(X: np.ndarray, Y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """把相邻把手(i,i+1)连成整板中心线段，并在两端各外延 0.275 m"""
    segs = []
    for i in range(len(X)-1):
        p_front = np.array([X[i],   Y[i]  ], float)
        p_back  = np.array([X[i+1], Y[i+1]], float)
        d = _unit(p_back - p_front)
        A = p_front - d * EXT_END
        B = p_back  + d * EXT_END
        segs.append((A, B))
    return segs

def rect_corners(A: np.ndarray, B: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    返回：矩形四点(顺时针)，中心坐标c，以及包围圆半径rad（用于宽相剔除）
    """
    d = _unit(B - A)
    n = np.array([-d[1], d[0]], float)
    hw = width / 2.0
    corners = np.vstack([A + n*hw, B + n*hw, B - n*hw, A - n*hw])
    c = 0.5*(A + B)
    L = _norm(B - A)               # 板中心线长度（≈ 3.41 或 2.20）
    rad = 0.5 * math.sqrt(L*L + width*width)   # 矩形的最小包围圆半径
    return corners, c, rad

def _project_interval(axis: np.ndarray, pts: np.ndarray) -> tuple[float, float]:
    a = axis / (_norm(axis) + 1e-15)
    s = pts @ a
    return float(s.min()), float(s.max())

def _sat_margin(c1: np.ndarray, c2: np.ndarray) -> float:
    """
    两个矩形的“带符号安全裕度”（与您现有 rect_pair_margin_SAT 一致）：
      >0 分离间隙；=0 接触；<0 重叠的负穿透
    """
    d1 = c1[1] - c1[0]; n1 = c1[3] - c1[0]
    d2 = c2[1] - c2[0]; n2 = c2[3] - c2[0]
    axes = [d1, n1, d2, n2]
    best_gap = -1e18
    min_overlap = 1e18
    for ax in axes:
        a1, b1 = _project_interval(ax, c1)
        a2, b2 = _project_interval(ax, c2)
        overlap = min(b1, b2) - max(a1, a2)
        if overlap < 0.0:
            gap = -overlap
            if gap > best_gap: best_gap = gap
        else:
            if overlap < min_overlap: min_overlap = overlap
    return best_gap if best_gap > 0.0 else -min_overlap

def min_clearance_for_time_b(t: float, b: float) -> float:
    """
    计算 μ(t) = 全队矩形对 (|i-j|>=2) 的最小安全裕度。
    加速：先用包围圆做宽相剔除（给出“保守间隙”），只有可能相交时才做 SAT。
    """
    X, Y = handles_at_time_b(t, b)
    segs = build_board_segments_from_handles(X, Y)
    rects, centers, radii = [], [], []
    for A, B in segs:
        rc, c, rad = rect_corners(A, B, W_BOARD)
        rects.append(rc); centers.append(c); radii.append(rad)
    centers = np.asarray(centers); radii = np.asarray(radii)

    mu = +1e9
    m = len(rects)
    for i in range(m):
        ci, ri = centers[i], radii[i]
        for j in range(i+2, m):  # 相邻忽略
            cj, rj = centers[j], radii[j]
            dist = _norm(cj - ci)
            circle_gap = dist - (ri + rj)
            if circle_gap >= 0.0:
                # 圆包围已分离；这给出一个“保守的正间隙”，可用于更新 μ
                if circle_gap < mu: mu = circle_gap
                continue
            # 否则做精确 SAT
            margin = _sat_margin(rects[i], rects[j])
            if margin < mu: mu = margin
            if mu < 0.0:        # 早停
                return mu
    return mu

# ====== 给定 p，计算“触边时刻” t_hit(p) ======
def t_hit_for_p(p: float) -> float | None:
    """
    触边角 θ_hit = R/b，时间满足：F(θ(t))=F(θ0)-t/b → t_hit = b*(F(θ0) - F(θ_hit))
    若初始半径 r0=16p < R，则无法“从外盘到边界”，返回 None。
    """
    if 16.0 * p < R_TURN:  # 物理下界：p < 0.28125 不满足
        return None
    b = p / (2.0 * math.pi)
    theta_hit = R_TURN / b
    return b * (F_theta(THETA0) - F_theta(theta_hit))

# ====== 在 [0, t_hit] 上扫描 μ(t)，粗到细，返回最小 μ 及其时刻 ======
def min_mu_over_path(p: float,
                     coarse_dt: float = 0.5,
                     refine_levels: tuple[float, ...] = (0.2, 0.1)) -> tuple[float, float]:
    """
    返回 (mu_min, t_at_min)。若中途发现 μ<0 则提前返回负值。
    """
    t_hit = t_hit_for_p(p)
    if t_hit is None:
        return -1.0, 0.0  # 视作不可行
    b = p / (2.0 * math.pi)

    # 1) 粗扫
    T = max(1, int(math.ceil(t_hit / coarse_dt)))
    t_grid = np.linspace(0.0, t_hit, T+1)
    mu_vals = []
    mu_min, t_min = +1e9, 0.0
    for t in t_grid:
        mu = min_clearance_for_time_b(t, b)
        mu_vals.append(mu)
        if mu < mu_min:
            mu_min, t_min = mu, t
        if mu < 0.0:
            return mu_min, t_min  # 早停：已不满足

    # 2) 逐级细化（在 t_min 附近对半缩小窗口）
    for dt in refine_levels:
        tL = max(0.0, t_min - 2.0*dt)
        tR = min(t_hit, t_min + 2.0*dt)
        n  = max(5, int((tR - tL) / dt) + 1)
        for t in np.linspace(tL, tR, n):
            mu = min_clearance_for_time_b(t, b)
            if mu < mu_min:
                mu_min, t_min = mu, t
            if mu < 0.0:
                return mu_min, t_min
    return mu_min, t_min

# ====== 判定给定 p 是否可行（全过程 μ>=0） ======
def is_feasible_pitch(p: float,
                      coarse_dt: float = 0.5) -> tuple[bool, float, float]:
    """
    返回 (feasible, mu_min, t_at_min)
    """
    mu_min, t_min = min_mu_over_path(p, coarse_dt=coarse_dt)
    return (mu_min >= 0.0), mu_min, t_min

# ====== 搜索最小可行螺距 p*（二分 + 回退惩罚） ======
def search_min_pitch(p_lo: float = 0.28125 + 1e-4,
                     p_hi_init: float = 0.8,
                     coarse_dt: float = 0.5,
                     tol: float = 1e-4,
                     max_expand: int = 12) -> dict:
    """
    先把上界扩到可行，再二分。若遇到罕见非单调，回退用惩罚目标 J(p)=p+λ*max(0,-μ_min)
    返回：{"p_star":..., "mu_min":..., "t_mu_min":..., "t_hit":..., "feasible":True/False}
    """
    # 扩上界：直到可行或超过上限
    p_hi = p_hi_init
    ok, _, _ = is_feasible_pitch(p_hi, coarse_dt=coarse_dt)
    expand_cnt = 0
    while not ok and expand_cnt < max_expand:
        p_hi *= 1.25
        ok, _, _ = is_feasible_pitch(p_hi, coarse_dt=coarse_dt)
        expand_cnt += 1
    if not ok:
        return {"p_star": None, "mu_min": None, "t_mu_min": None,
                "t_hit": None, "feasible": False, "msg": "未找到可行上界，请增大 p_hi_init 或放宽 coarse_dt"}

    # 二分搜索（假设“可行性随 p 近似单调”）
    pL, pR = p_lo, p_hi
    best = {"p_star": pR, "mu_min": +1e9, "t_mu_min": 0.0, "t_hit": t_hit_for_p(pR), "feasible": True}
    for _ in range(50):
        if pR - pL < tol:
            break
        pm = 0.5*(pL + pR)
        feas, mu_min, t_min = is_feasible_pitch(pm, coarse_dt=coarse_dt)
        if feas:
            # 更新当前最优
            best.update({"p_star": pm, "mu_min": mu_min, "t_mu_min": t_min, "t_hit": t_hit_for_p(pm), "feasible": True})
            pR = pm
        else:
            pL = pm

    # 轻微非单调时的回退：用惩罚式做一次邻域局部搜
    # J(p)=p + λ*max(0,-μ_min)，λ 取 10（可调）
    lam = 10.0
    for trial in np.linspace(max(p_lo, best["p_star"] - 5e-3), best["p_star"] + 5e-3, 7):
        feas, mu_min, t_min = is_feasible_pitch(trial, coarse_dt=coarse_dt)
        J = trial + lam * max(0.0, -mu_min)
        J_best = best["p_star"] + lam * max(0.0, -best["mu_min"])
        if J < J_best:
            best.update({"p_star": trial, "mu_min": mu_min, "t_mu_min": t_min, "t_hit": t_hit_for_p(trial), "feasible": feas})

    return best

# ================== 使用示例（需你手动运行） ==================
if __name__ == "__main__":
    # 你可以根据机器性能调小 coarse_dt 加精度（会更慢），或调大以提速
    result = search_min_pitch(p_lo=0.28125 + 1e-4, p_hi_init=0.8, coarse_dt=0.5, tol=1e-4)
    print("[Q3] 搜索结果：", result)
    # result["p_star"] 即“第三问（含不碰撞约束）的最小螺距”估计值
