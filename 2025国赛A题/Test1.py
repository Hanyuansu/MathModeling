# ---------- L1：全部采样点的“球心垂线”示意 ----------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def _pick_focus_time_longest(intervals, t_burst, dt=0.01):
    """选最长遮蔽段中点；若无遮蔽，则取有效窗中点"""
    if intervals:
        a, b = max(intervals, key=lambda ab: ab[1]-ab[0])
        return 0.5*(a+b)
    T_HIT_loc = missile_hit_time(M0)
    t0, t1 = t_burst, min(t_burst+T_EFFECT, T_HIT_loc)
    return 0.5*(t0+t1)

def _plot_cylinder(ax, color='#777', alpha=0.12, n_ang=80, n_z=40):
    cx, cy, z0 = CYL_CENTER
    zs = np.linspace(0.0, H_TAR, n_z) + z0
    ang = np.linspace(0, 2*np.pi, n_ang)
    X = cx + R_TAR*np.cos(ang)[:,None]*np.ones_like(zs)[None,:]
    Y = cy + R_TAR*np.sin(ang)[:,None]*np.ones_like(zs)[None,:]
    Z = np.ones_like(ang)[:,None]*zs[None,:]
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.25,
                    color=color, alpha=alpha, edgecolor='k')

def plot_L1_perpmap_all(ans,
                        N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
                        show_cylinder=True, save_path=None,
                        color_in='tab:green', color_out='#bbbbbb',
                        lw_in=1.8, lw_out=1.0, alpha_out=0.65):
    """
    L1示意：对全部采样点，作“球心到视轴(采样点→导弹)的垂线”，
    距离<=R_SMOKE 的垂线高亮（在烟雾范围内）。
    """
    # 读取解
    th = math.radians(float(ans["theta_deg"]))
    v  = float(ans["v_u_mps"])
    td = float(ans["t_drop_s"])
    ta = float(ans["tau_s"])
    tb = td + ta
    intervals = ans["cover_intervals_s"]

    # 时刻与位置
    s_burst = burst_point(U0, th, v, td, ta)
    t_focus = _pick_focus_time_longest(intervals, tb, dt=0.01)
    m = missile_pos(M0, t_focus)
    s = smoke_center_after_burst(s_burst, t_focus, tb)

    # L1 采样点
    PTS = build_cylinder_samples(N_ang=N_ANG, N_z=N_Z, include_side=INCLUDE_SIDE)

    # 计算所有垂线段（s -> Y），区分是否在球内
    lines_in, lines_out = [], []
    for P in PTS:
        vseg   = m - P
        vv     = float(np.dot(vseg, vseg))
        if vv <= 0.0:
            continue
        alpha_raw = float(np.dot(s - P, vseg) / vv)  # 到“无限直线”的参数
        alpha     = 0.0 if alpha_raw < 0.0 else (1.0 if alpha_raw > 1.0 else alpha_raw)  # 夹到线段上
        Y = P + alpha * vseg
        d = float(np.linalg.norm(s - Y))
        if d <= R_SMOKE + 1e-9:
            lines_in.append((s.copy(), Y))
        else:
            lines_out.append((s.copy(), Y))

    # 作图
    fig = plt.figure(figsize=(9.6, 6.9), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title("L1 遮掩示意图（全部采样点的球心垂线）")

    # 可选：圆柱
    if show_cylinder:
        _plot_cylinder(ax, color='#777', alpha=0.12, n_ang=80, n_z=40)

    # 完整烟球
    ugrid = np.linspace(0, 2*np.pi, 90)
    vgrid = np.linspace(0,   np.pi, 45)
    xs = s[0] + R_SMOKE*np.outer(np.cos(ugrid), np.sin(vgrid))
    ys = s[1] + R_SMOKE*np.outer(np.sin(ugrid), np.sin(vgrid))
    zs = s[2] + R_SMOKE*np.outer(np.ones_like(ugrid), np.cos(vgrid))
    ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, linewidth=0.25,
                    alpha=0.25, color='tab:red', edgecolor='k')

    # 垂线集合（球内/球外）
    if lines_out:
        coll_out = Line3DCollection([np.vstack(l) for l in lines_out],
                                    colors=color_out, linewidths=lw_out, alpha=alpha_out)
        ax.add_collection3d(coll_out)
    if lines_in:
        coll_in = Line3DCollection([np.vstack(l) for l in lines_in],
                                   colors=color_in, linewidths=lw_in, alpha=0.95)
        ax.add_collection3d(coll_in)

    # 球心
    ax.scatter([s[0]], [s[1]], [s[2]], c='tab:red', s=28, label='球心')

    # 视域：以球心为中心，放大到能看清所有垂足（但不铺太大）
    all_pts = [s] + [y for _, y in lines_in] + [y for _, y in lines_out]
    A = np.array(all_pts)
    span = np.max(np.abs(A - s), axis=0) if len(all_pts) > 1 else np.array([R_SMOKE, R_SMOKE, R_SMOKE])
    half = float(max(1.6*R_SMOKE, np.max(span) + 0.5*R_SMOKE))
    ax.set_xlim(s[0]-half, s[0]+half)
    ax.set_ylim(s[1]-half, s[1]+half)
    ax.set_zlim(s[2]-half, s[2]+half)
    ax.set_box_aspect((1,1,1))
    ax.set_xlabel('X / m'); ax.set_ylabel('Y / m'); ax.set_zlabel('Z / m')
    ax.view_init(elev=22, azim=-55)

    # 图例（用虚拟句柄）
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color='tab:red', lw=6, alpha=0.25, label='烟雾球'),
        Line2D([0],[0], color=color_in, lw=2.4, label='垂线（落在烟雾内）'),
        Line2D([0],[0], color=color_out, lw=1.6, alpha=alpha_out, label='垂线（烟雾外）'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='tab:red', markersize=6, label='球心')
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=9)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=180)
        print(f"[OK] 图已保存：{save_path}")
    plt.show()

    plot_L1_perpmap_all(ans_L1,
                        N_ANG=48, N_Z=9, INCLUDE_SIDE=True,
                        show_cylinder=True,
                        save_path="result/Problem2_result/Q2_L1.png")