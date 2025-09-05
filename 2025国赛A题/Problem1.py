# -*- coding: utf-8 -*-
"""

"""

import os
import math
import numpy as np


R_TAR, H_TAR = 7.0, 10.0

# 初始位置（题面给定）
T_DROP = 1.5
TAU = 3.6
T_BURST = T_DROP + TAU

# 时间窗口
T0 = T_BURST
T1 = T_BURST + 20.0
DT = 0.001

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def missile_pos(t: float) -> np.ndarray:
    return M1_0 + VM * d * t

def uav_pos(t: float) -> np.ndarray:
    return np.array([FY1_0[0] + VU * HEADING[0] * t,
                     FY1_0[1] + VU * HEADING[1] * t,
                     FY1_0[2]])

def burst_point() -> np.ndarray:
    r_drop = uav_pos(T_DROP)
    horiz = np.array([VU * HEADING[0] * TAU, VU * HEADING[1] * TAU, 0.0])
    vert  = np.array([0.0, 0.0, -0.5 * g * TAU * TAU])
    return r_drop + horiz + vert

S_BURST = burst_point()

def smoke_center(t: float) -> np.ndarray:
    dz = -V_SINK * max(0.0, t - T_BURST)
    return S_BURST + np.array([0.0, 0.0, dz])

def point_seg_distance(P: np.ndarray, Q: np.ndarray, X: np.ndarray) -> float:
    """点 X 到线段 PQ 的最小距离"""
    v = Q - P
    vv = float(np.dot(v, v))
        return float(np.linalg.norm(X - P))
    a = float(np.dot(X - P, v) / vv)
    Y = P + a * v
    return float(np.linalg.norm(X - Y))

def covered_L0(t: float) -> bool:
    m = missile_pos(t)
    s = smoke_center(t)
    return point_seg_distance(P_CENTER, m, s) <= R_SMOKE

    while t <= t1 + 1e-12:
        if flag and not in_seg:
        if (not flag) and in_seg:
            in_seg = False
            intervals.append((seg_start, t))
        t += dt
    if in_seg:
        intervals.append((seg_start, t1))

# ------------------- 主程序 -------------------
if __name__ == "__main__":
    print(f"Burst time te = {T_BURST:.3f} s")
    print(f"Burst point  = ({S_BURST[0]:.3f}, {S_BURST[1]:.3f}, {S_BURST[2]:.3f}) m")
