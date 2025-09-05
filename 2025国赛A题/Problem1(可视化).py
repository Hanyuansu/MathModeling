# -*- coding: utf-8 -*-
"""
CUMCM 2025 A题 第1问：遮蔽“单帧”可视化（只画遮蔽瞬间）
要素：目标圆柱、烟球位置（球心+球体）、导弹—原点连线（不画导弹点）、被遮蔽的圆柱表面范围

输出：result/Q1_instant.png
"""

import os
import math
import numpy as np

# ------------------- 题面常量（与原题一致） -------------------
g = 9.81
VM = 300.0
V_SINK = 3.0
R_SMOKE = 10.0
T_EFFECT = 20.0

# 目标圆柱参数（仅用几何中心点，不画圆柱体）
R_TAR, H_TAR = 7.0, 10.0
CYL_CENTER = np.array([0.0, 200.0, 0.0])         # 下底面圆心
P_CENTER  = np.array([0.0, 200.0, H_TAR/2.0])    # 圆柱几何中心（L0 用）

# 初始位置（题面给定）
M1_0 = np.array([20000.0, 0.0, 2000.0])          # 导弹初始
FY1_0 = np.array([17800.0, 0.0, 1800.0])         # 无人机初始
VU = 120.0
HEADING = np.array([-1.0, 0.0])
T_DROP = 1.5
TAU = 3.6
T_BURST = T_DROP + TAU

# 时间窗口
T0 = T_BURST
T1 = T_BURST + 20.0
DT = 0.001

# ------------------- 运动学函数 -------------------
def unit(v: np.ndarray) -> np.ndarray:
    """单位向量；零向量则原样返回"""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def missile_pos(t: float) -> np.ndarray:
    """导弹：恒速直线朝原点飞行（仅用于计算方向/相交）"""
    d = unit(-M1_0)        # 指向原点的方向
    return M1_0 + VM * d * t

def uav_pos(t: float) -> np.ndarray:
    """无人机：等高匀速，仅用于起爆点推导"""
    return np.array([FY1_0[0] + VU * HEADING[0] * t,
                     FY1_0[1] + VU * HEADING[1] * t,
                     FY1_0[2]])

def burst_point() -> np.ndarray:
    """起爆点：投放后 TAU 秒；水平继承无人机速度，竖直自由落体"""
    r_drop = uav_pos(T_DROP)
    horiz = np.array([VU * HEADING[0] * TAU, VU * HEADING[1] * TAU, 0.0])
    vert  = np.array([0.0, 0.0, -0.5 * g * TAU * TAU])
    return r_drop + horiz + vert

S_BURST = burst_point()

def smoke_center(t: float) -> np.ndarray:
    """烟球球心：起爆后以 3 m/s 均速下沉"""
    dz = -V_SINK * max(0.0, t - T_BURST)
    return S_BURST + np.array([0.0, 0.0, dz])

# ------------------- L0 覆盖判定 -------------------
def point_seg_distance(P: np.ndarray, Q: np.ndarray, X: np.ndarray) -> float:
    """点 X 到线段 PQ 的最小距离"""
    v = Q - P
    vv = float(np.dot(v, v))
    if vv <= 0.0:
        return float(np.linalg.norm(X - P))
    a = float(np.dot(X - P, v) / vv)
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    Y = P + a * v
    return float(np.linalg.norm(X - Y))

def covered_L0(t: float) -> bool:
    """L0：用 P_CENTER 与 M(t) 的线段，判定与半径 R_SMOKE 的球是否相交"""
    m = missile_pos(t)
    s = smoke_center(t)
    return point_seg_distance(P_CENTER, m, s) <= R_SMOKE

def find_intervals_L0(t0: float, t1: float, dt: float):
    """扫描时间轴，返回 L0 遮蔽区间列表 [(a,b), ...]"""
    intervals, t = [], t0
    in_seg, seg_start = False, None
    while t <= t1 + 1e-12:
        flag = covered_L0(t)
        if flag and not in_seg:
            in_seg, seg_start = True, t
        if (not flag) and in_seg:
            in_seg = False
            intervals.append((seg_start, t))
        t += dt
    if in_seg:
        intervals.append((seg_start, t1))
    return intervals

# ------------------- 可视化（L0 单帧） -------------------
def plot_L0_instant(save_path: str):
    """
    1) 计算 L0 遮蔽区间并选取 t_focus（第一段中点）
    2) 画“遮蔽时刻的球心轨迹”（红色粗线）
    3) 在 t_focus 画半透明球体 + 在球内画一小段“导弹→圆柱中心”的连线（方向正确且与球相交）
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    # 计算遮蔽区间
    intervals = find_intervals_L0(T0, T1, DT)
    ts = np.arange(T0, T1 + 1e-12, DT)
    flags = np.array([covered_L0(t) for t in ts], dtype=bool)
    S_track = np.stack([smoke_center(t) for t in ts], axis=0)

    if len(intervals) == 0:
        # 避免极端情况：无遮蔽则取最近时刻，仍给出单帧示意
        print("[WARN] 本窗口内 L0 无遮蔽，自动选择最近距离时刻。")
        best_t, best_d = T0, 1e18
        for tt in ts:
            d = point_seg_distance(P_CENTER, missile_pos(tt), smoke_center(tt))
            if d < best_d:
                best_d, best_t = d, tt
        t_focus = best_t
    else:
        a0, b0 = intervals[0]
        t_focus = 0.5 * (a0 + b0)
        print(f"[L0] 遮蔽区间：{[(round(a,3), round(b,3)) for a,b in intervals]}  ->  展示时刻 t={t_focus:.3f}s")

    # 焦点时刻的关键量
    m = missile_pos(t_focus)
    s = smoke_center(t_focus)

    # —— 计算“导弹→圆柱中心”连线在球内的短线段（只画球内段）——
    # 以 P_CENTER 为参考点，方向取指向“导弹”的单位向量
    u = unit(m - P_CENTER)                 # 方向（P_CENTER -> 导弹）
    w = P_CENTER - s                       # 线参考点相对球心的向量
    # 求解 |w + u t|^2 = R^2
    b = 2.0 * float(np.dot(u, w))
    c = float(np.dot(w, w)) - R_SMOKE**2
    disc = b*b - 4.0*c
    seg_inside = None
    if disc >= 0.0:
        sqrtD = math.sqrt(disc)
        t1 = (-b - sqrtD) / 2.0
        t2 = (-b + sqrtD) / 2.0
        t_entry, t_exit = (t1, t2) if t1 <= t2 else (t2, t1)
        P_in  = P_CENTER + u * t_entry
        P_out = P_CENTER + u * t_exit
        seg_inside = (P_in, P_out)         # 球内的短线段

    # —— 作图 ——
    fig = plt.figure(figsize=(8.4, 6.8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title(f"L0 单帧示意：t = {t_focus:.3f} s（只画遮蔽轨迹 + 球体 + 短线段）")

    # 1) 画出“发生遮蔽时刻”的烟球轨迹（红色粗线）
    if flags.any():
        ax.plot(S_track[flags,0], S_track[flags,1], S_track[flags,2],
                color='tab:red', lw=2.8, label='遮蔽时刻的球心轨迹(L0)')

    # 2) 焦点时刻的烟球（半透明球体 + 球心）
    ugrid = np.linspace(0, 2*np.pi, 50)
    vgrid = np.linspace(0,   np.pi, 26)
    xs = s[0] + R_SMOKE*np.outer(np.cos(ugrid), np.sin(vgrid))
    ys = s[1] + R_SMOKE*np.outer(np.sin(ugrid), np.sin(vgrid))
    zs = s[2] + R_SMOKE*np.outer(np.ones_like(ugrid), np.cos(vgrid))
    ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, linewidth=0.2,
                    alpha=0.25, color='tab:red', edgecolor='k')
    ax.scatter([s[0]],[s[1]],[s[2]], c='tab:red', s=30, label='球心')

    # 3) 在球内画一小段“导弹→圆柱中心”的连线（方向正确、与球相交且不过分长）
    if seg_inside is not None:
        P_in, P_out = seg_inside
        ax.plot([P_in[0], P_out[0]], [P_in[1], P_out[1]], [P_in[2], P_out[2]],
                'k--', lw=2.0, label='导弹→圆柱中心（球内短段）')
        # 在入口端给一个小箭头，强调方向（P_CENTER→导弹）
        arrow_len = 0.6 * R_SMOKE
        a_base = P_in
        a_tip  = P_in + u * arrow_len
        ax.plot([a_base[0], a_tip[0]], [a_base[1], a_tip[1]], [a_base[2], a_tip[2]],
                color='k', lw=2.2)
        ax.scatter([a_tip[0]], [a_tip[1]], [a_tip[2]], c='k', s=18)

    # —— 轴域：只围绕“遮蔽轨迹 + 球体 + 短线段”，保持紧凑 ——
    xs_all = [s[0]-R_SMOKE, s[0]+R_SMOKE]
    ys_all = [s[1]-R_SMOKE, s[1]+R_SMOKE]
    zs_all = [s[2]-R_SMOKE, s[2]+R_SMOKE]
    if flags.any():
        xs_all += [S_track[flags,0].min(), S_track[flags,0].max()]
        ys_all += [S_track[flags,1].min(), S_track[flags,1].max()]
        zs_all += [S_track[flags,2].min(), S_track[flags,2].max()]
    if seg_inside is not None:
        xs_all += [P_in[0], P_out[0]]
        ys_all += [P_in[1], P_out[1]]
        zs_all += [P_in[2], P_out[2]]

    x_min, x_max = min(xs_all), max(xs_all)
    y_min, y_max = min(ys_all), max(ys_all)
    z_min, z_max = min(zs_all), max(zs_all)
    pad = 0.25 * R_SMOKE
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_zlim(z_min - pad, z_max + pad)
    ax.set_box_aspect((1,1,1))
    ax.set_xlabel('X / m'); ax.set_ylabel('Y / m'); ax.set_zlabel('Z / m')
    ax.view_init(elev=25, azim=-60)
    ax.legend(loc='best', fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180)
    print(f"[OK] 已保存：{save_path}")
    plt.show()

# ------------------- 主程序 -------------------
if __name__ == "__main__":
    print(f"Burst time te = {T_BURST:.3f} s")
    print(f"Burst point  = ({S_BURST[0]:.3f}, {S_BURST[1]:.3f}, {S_BURST[2]:.3f}) m")
    out_path = os.path.join("result", "Q1_L0_instant.png")
    plot_L0_instant(out_path)