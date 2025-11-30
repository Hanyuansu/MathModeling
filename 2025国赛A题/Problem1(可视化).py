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
P_CENTER   = np.array([0.0, 200.0, H_TAR/2.0])

M1_0  = np.array([20000.0, 0.0, 2000.0])
FY1_0 = np.array([17800.0, 0.0, 1800.0])
VU       = 120.0
HEADING  = np.array([-1.0, 0.0])
T_DROP   = 1.5
TAU      = 3.6
T_BURST  = T_DROP + TAU

T0, T1 = T_BURST, T_BURST + 20.0
DT = 0.001

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def missile_pos(t):
    d = unit(-M1_0)
    return M1_0 + VM * d * t

def uav_pos(t):
    return np.array([FY1_0[0] + VU * HEADING[0] * t,
                     FY1_0[1] + VU * HEADING[1] * t,
                     FY1_0[2]])

def burst_point():
    r_drop = uav_pos(T_DROP)
    horiz = np.array([VU * HEADING[0] * TAU, VU * HEADING[1] * TAU, 0.0])
    vert  = np.array([0.0, 0.0, -0.5 * g * TAU * TAU])
    return r_drop + horiz + vert

S_BURST = burst_point()

def smoke_center(t):
    dz = -V_SINK * max(0.0, t - T_BURST)
    return S_BURST + np.array([0.0, 0.0, dz])

def point_seg_distance(P, Q, X):
    v = Q - P
    vv = float(np.dot(v, v))
    if vv <= 0.0:
        return float(np.linalg.norm(X - P))
    a = float(np.dot(X - P, v) / vv)
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    Y = P + a * v
    return float(np.linalg.norm(X - Y))

def covered_L0(t):
    m = missile_pos(t)
    s = smoke_center(t)
    return point_seg_distance(P_CENTER, m, s) <= R_SMOKE

def find_intervals_L0(t0, t1, dt):
    intervals, t = [], t0
    in_seg, a = False, None
    while t <= t1 + 1e-12:
        f = covered_L0(t)
        if f and not in_seg:
            in_seg, a = True, t
        if (not f) and in_seg:
            in_seg = False
            intervals.append((a, t))
        t += dt
    if in_seg:
        intervals.append((a, t1))
    return intervals

def plot_cover_schematic(save_path=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    ts = np.arange(T0, T1 + 1e-12, DT)
    intervals = find_intervals_L0(T0, T1, DT)
    if intervals:
        a0, b0 = intervals[0]
        t_focus = 0.5*(a0+b0)
    else:
        best_t, best_d = T0, 1e18
        for tt in ts:
            d = point_seg_distance(P_CENTER, missile_pos(tt), smoke_center(tt))
            if d < best_d:
                best_d, best_t = d, tt
        t_focus = best_t

    m = missile_pos(t_focus)
    s = smoke_center(t_focus)
    u = unit(m - P_CENTER)

    w = P_CENTER - s
    b = 2.0 * float(np.dot(u, w))
    c = float(np.dot(w, w)) - R_SMOKE**2
    disc = b*b - 4.0*c
    have_intersection = disc >= 0.0
    if have_intersection:
        sqrtD = math.sqrt(disc)
        t1 = (-b - sqrtD) / 2.0
        t2 = (-b + sqrtD) / 2.0
        if t1 > t2: t1, t2 = t2, t1
        P_in  = P_CENTER + u * t1
        P_out = P_CENTER + u * t2

    t_foot = float(np.dot(s - P_CENTER, u))
    P_foot = P_CENTER + u * t_foot

    fig = plt.figure(figsize=(8.6, 6.6), constrained_layout=True)
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.set_title("遮掩示意图")

    ugrid = np.linspace(0, 2*np.pi, 80)
    vgrid = np.linspace(0,   np.pi, 40)
    xs = s[0] + R_SMOKE*np.outer(np.cos(ugrid), np.sin(vgrid))
    ys = s[1] + R_SMOKE*np.outer(np.sin(ugrid), np.sin(vgrid))
    zs = s[2] + R_SMOKE*np.outer(np.ones_like(ugrid), np.cos(vgrid))
    ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, linewidth=0.3,
                    alpha=0.25, color='tab:red', edgecolor='k')

    L = 2.2 * R_SMOKE
    if have_intersection:
        ax.plot([P_in[0], P_out[0]],[P_in[1], P_out[1]],[P_in[2], P_out[2]],
                'k-', lw=2.2, label='导弹视轴（球内实线）')
        L1 = P_in - u*L
        ax.plot([L1[0], P_in[0]],[L1[1], P_in[1]],[L1[2], P_in[2]],
                'k--', lw=2.2)
        L2 = P_out + u*L
        ax.plot([P_out[0], L2[0]],[P_out[1], L2[1]],[P_out[2], L2[2]],
                'k--', lw=2.2)
    else:
        O1 = P_CENTER - u*L; O2 = P_CENTER + u*L
        ax.plot([O1[0],O2[0]],[O1[1],O2[1]],[O1[2],O2[2]], 'k--', lw=2.2,
                label='导弹视轴')

    ax.scatter([s[0]],[s[1]],[s[2]], c='tab:red', s=28, label='球心')
    ax.scatter([P_foot[0]],[P_foot[1]],[P_foot[2]], c='k', s=28, label='垂足 P')
    ax.plot([s[0], P_foot[0]],[s[1], P_foot[1]],[s[2], P_foot[2]],
            color='tab:red', lw=2.0)

    pts = [s, P_foot]
    if have_intersection:
        pts += [P_in, P_out]
    pts = np.array(pts)
    max_dev = float(np.max(np.abs(pts - s)))
    half = max(1.35*R_SMOKE, max_dev + 0.35*R_SMOKE)
    ax.set_xlim(s[0]-half, s[0]+half)
    ax.set_ylim(s[1]-half, s[1]+half)
    ax.set_zlim(s[2]-half, s[2]+half)
    ax.set_box_aspect((1,1,1))   # 关键：三轴同比例
    ax.set_xlabel('X / m'); ax.set_ylabel('Y / m'); ax.set_zlabel('Z / m')
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc='upper right', fontsize=9)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=180)
        print(f"[OK] 已保存：{save_path}")
    plt.show()

def plot_cover_timeline(save_path=None, show_intervals_text=False, show_total=True):

    import matplotlib.pyplot as plt

    intervals = find_intervals_L0(T0, T1, DT)
    total_cover = sum(b - a for (a, b) in intervals)

    fig, ax = plt.subplots(figsize=(8.6, 2.0), constrained_layout=True)
    ax.hlines(1, T0, T1, color="#dddddd", lw=10, label="Valid Window")
    for (a, b) in intervals:
        ax.hlines(1, a, b, color="tab:red", lw=10, label="Covering Section")

    ax.set_ylim(0.8, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("t / s")

    title = "M1  Total obscuration duration=1.496 s"
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=9)


    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=180)
        print(f"[OK] 已保存：{save_path}")
    plt.show()



if __name__ == "__main__":
    out_path = os.path.join("result/Problem1_result", "Q1.png")
    plot_cover_schematic(out_path)
    out_path2 = os.path.join("result/Problem1_result", "遮掩时长.png")
    plot_cover_timeline(out_path2)
