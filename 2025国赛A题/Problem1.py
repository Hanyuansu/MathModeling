import os
import math
import numpy as np

g = 9.81
VM = 300.0
V_SINK = 3.0
R_SMOKE = 10.0
T_EFFECT = 20.0

R_TAR, H_TAR = 7.0, 10.0
CYL_CENTER = np.array([0.0, 200.0, 0.0])
P_CENTER  = np.array([0.0, 200.0, H_TAR/2.0])

M1_0 = np.array([20000.0, 0.0, 2000.0])
FY1_0 = np.array([17800.0, 0.0, 1800.0])

VU = 120.0
HEADING = np.array([-1.0, 0.0])
T_DROP = 1.5
TAU = 3.6
T_BURST = T_DROP + TAU

T0 = T_BURST
T1 = T_BURST + 20.0
DT = 0.001

USE_SIDE = True
USE_L0 = True

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def missile_pos(t: float) -> np.ndarray:
    d = unit(-M1_0)
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

# ------------------- 距离与采样 -------------------
def point_seg_distance(P: np.ndarray, Q: np.ndarray, X: np.ndarray) -> float:
    v = Q - P
    vv = float(np.dot(v, v))
    if vv == 0.0:
        return float(np.linalg.norm(X - P))
    a = float(np.dot(X - P, v) / vv)
    a = 0.0 if a < 0 else (1.0 if a > 1.0 else a)
    Y = P + a * v
    return float(np.linalg.norm(X - Y))

def cyl_points_top_bottom(N_ang: int = 72) -> np.ndarray:
    cx, cy, cz = CYL_CENTER
    out = []
    for z in (cz, cz + H_TAR):
        for k in range(N_ang):
            ang = 2.0 * math.pi * k / N_ang
            x = cx + R_TAR * math.cos(ang)
            y = cy + R_TAR * math.sin(ang)
            out.append(np.array([x, y, z]))
    return np.stack(out, axis=0)

def cyl_points_side(N_ang: int = 72, N_z: int = 11) -> np.ndarray:
    cx, cy, cz = CYL_CENTER
    zs = np.linspace(cz, cz + H_TAR, N_z)
    out = []
    for z in zs:
        for k in range(N_ang):
            ang = 2.0 * math.pi * k / N_ang
            x = cx + R_TAR * math.cos(ang)
            y = cy + R_TAR * math.sin(ang)
            out.append(np.array([x, y, z]))
    return np.stack(out, axis=0)

PTS_L1 = [cyl_points_top_bottom(72)]
if USE_SIDE:
    PTS_L1.append(cyl_points_side(72, 11))
PTS_L1 = np.concatenate(PTS_L1, axis=0)

def covered_L0(t: float) -> bool:
    m = missile_pos(t)
    s = smoke_center(t)
    return point_seg_distance(P_CENTER, m, s) <= R_SMOKE

def covered_L1(t: float) -> bool:
    m = missile_pos(t)
    s = smoke_center(t)
    for p in PTS_L1:
        if point_seg_distance(p, m, s) <= R_SMOKE:
            return True
    return False

def integrate_cover(flag_func, t0: float, t1: float, dt: float):
    covered = 0.0
    intervals = []
    t = t0
    in_seg = False
    seg_start = None
    while t <= t1 + 1e-12:
        flag = flag_func(t)
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

if __name__ == "__main__":
    print(f"Burst time te = {T_BURST:.3f} s")
    print(f"Burst point  = ({S_BURST[0]:.3f}, {S_BURST[1]:.3f}, {S_BURST[2]:.3f}) m")

    cov_L1, segs_L1 = integrate_cover(covered_L1, T0, T1, DT)
    print(f"[L1] cover time = {cov_L1:.3f} s, intervals = {[(round(a,3), round(b,3)) for a,b in segs_L1]}")
    if USE_L0:
        cov_L0, segs_L0 = integrate_cover(covered_L0, T0, T1, DT)
        print(f"[L0] cover time = {cov_L0:.3f} s, intervals = {[(round(a,3), round(b,3)) for a,b in segs_L0]}")



