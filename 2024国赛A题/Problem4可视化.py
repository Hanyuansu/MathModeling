# -*- coding: utf-8 -*-
"""
dragon2.py
用途：第四问/双圆弧调头（先大弧后小弧，2:1）+ 螺线盘出（含调头前的盘入），
      复现你给的“dragon2.py”截图里的思路与结构。

路径编号（与截图一致）：
  1 = 盘入等距螺线  r = b * θ
  2 = 大圆弧（B→D）
  3 = 小圆弧（D→F）
  4 = 盘出等距螺线  r = b * (θ - π)

核心接口：
  Dragon.set_time(t, need_vel=False)    # 设定时间并计算全队位置；t=0 在 B（触边）
  Dragon.print_status()                  # 画示意图自检（圆心、圆弧、触点、把手散点）

说明：
- 头把手恒速 1 m/s 沿整条拼接路径运动。t=0 在触边点 B；t<0 在盘入螺线；t>0 先大弧、再小弧、再盘出螺线。
- “逐节回推”时，优先用同段的解析公式（与截图一致）；若需跨段，则先在当前段吃掉能吃的，再把“剩余弦长”交给上一个段继续算。
- 同段在螺线上的“角度增量”用牛顿 + 二分稳健求。
"""

from __future__ import annotations
import math
from typing import Tuple, List
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# ============= 全局常量（与截图一致） =============
ben_wid = 0.30        # 板宽
ben_len_head = 3.41   # 龙头板长
ben_len = 2.20        # 龙身板长
hole_dis = 0.275      # 孔心到端部（绘图时可用，本文件未用）
hole_dis2 = 0.055     # 孔径（本文件未用）
dra_len = 223         # 板凳个数（223块）

# 孔心距（前把手→后把手）
L_head = ben_len_head
L_body = ben_len
D_hole = 0.55
l_HEAD = L_head - D_hole   # 2.86
l_BODY = L_body - D_hole   # 1.65
l_TAIL = l_BODY            # 1.65

# 螺距与掉头圆
d = 1.7                    # 螺距 p
r_turn = 4.5               # 掉头圆半径 R_turn
b = d / (2.0 * math.pi)    # 等距螺线系数 r = b θ

# 触边角（截图：theta = 2*pi*r/d）
theta_B = 2.0 * math.pi * r_turn / d              # = R_turn / b
alpha = math.atan(theta_B)                        # 截图里的 “圆中 alpha 角”
R1 = 3.0 / math.sin(alpha)                        # 大圆半径（≈ 3/sinα）
R2 = 3.0 / (2.0 * math.sin(alpha))                # 小圆半径（≈ 1.5/sinα），保证 R1:R2 = 2:1

# ---- 两圆圆心坐标（截图公式） ----
# B 点极角为 theta_B，坐标：
Bx = r_turn * math.cos(theta_B)
By = r_turn * math.sin(theta_B)
# F = -B
Fx, Fy = -Bx, -By

# 圆心（截图 41~46 行）
O1x = r_turn * math.cos(theta_B) - R1 * math.sin(theta_B + alpha)
O1y = r_turn * math.sin(theta_B) + R1 * math.cos(theta_B + alpha)
O2x = -r_turn * math.cos(theta_B) + R2 * math.sin(theta_B + alpha)
O2y = -r_turn * math.sin(theta_B) - R2 * math.cos(theta_B + alpha)
C1 = np.array([O1x, O1y], float)   # 大圆心
C2 = np.array([O2x, O2y], float)   # 小圆心

# 圆弧方向（由切线方向决定）。盘入为顺时针、盘出为逆时针；
# 对应在圆上：B 处切向沿“路径前进方向”应与圆上单位切向同向。
def rot90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], float)

def spiral_tangent_in(theta: float) -> np.ndarray:
    # dP/dθ（不归一），盘入沿运动方向取负号（θ 随时间减小）
    dx = b*math.cos(theta) - b*theta*math.sin(theta)
    dy = b*math.sin(theta) + b*theta*math.cos(theta)
    v = np.array([-dx, -dy], float)  # 顺时针
    n = np.hypot(v[0], v[1]);  return v/(n if n>1e-15 else 1.0)

def spiral_tangent_out(theta: float) -> np.ndarray:
    # r = b(θ-π)
    u = theta - math.pi
    dx = b*math.cos(theta) - b*u*math.sin(theta)
    dy = b*math.sin(theta) + b*u*math.cos(theta)
    v = np.array([dx, dy], float)    # 逆时针
    n = np.hypot(v[0], v[1]);  return v/(n if n>1e-15 else 1.0)

tB = spiral_tangent_in(theta_B)
tF = spiral_tangent_out(theta_B + math.pi)

# 大圆从 B 开始，方向由 <Rot90(B-C1), tB> 确定
def angle_of(v):
    return math.atan2(v[1], v[0])

uB = np.array([Bx, By]) - C1
ang_B = angle_of(uB)

# 圆在 B 点的 CCW 单位切向
tan_ccw_at_B = rot90(uB) / np.hypot(*uB)
# 盘入在 B 点的真实前进切向
tB = spiral_tangent_in(theta_B)

# 先按切向一致性猜测方向
sgn1 = +1 if np.dot(tan_ccw_at_B, tB) > 0 else -1

# >>> FIX: 保险校验——若用这个 sgn1 得到的“圆上前进切向”仍与 tB 夹钝角，就翻转
t_arc_fwd_at_B = sgn1 * tan_ccw_at_B
if np.dot(t_arc_fwd_at_B, tB) < 0:   # 与螺线切向反了
    sgn1 *= -1

# D 点仍按外切点（两圆连心线），R1:R2=2:1 ⇒ D=C1+2/3(C2-C1)
Dxy = C1 + (R1/(R1+R2)) * (C2 - C1)

# 小圆方向同理在 F 点做一次“点火校验”
ang_D = angle_of(Dxy - C2)
uF = np.array([Fx, Fy]) - C2
tan_ccw_at_F = rot90(uF) / np.hypot(*uF)
tF = spiral_tangent_out(theta_B + math.pi)

sgn2 = +1 if np.dot(tan_ccw_at_F, tF) > 0 else -1
# >>> FIX: 同样做一次保险校验
t_arc_fwd_at_F = sgn2 * tan_ccw_at_F
if np.dot(t_arc_fwd_at_F, tF) < 0:
    sgn2 *= -1

# 有向圆心角（注意使用刚确定好的 sgn1/sgn2）
def ang_diff(a_to: float, a_from: float, ccw: bool) -> float:
    d = (a_to - a_from) % (2*math.pi)
    return d if ccw else ((a_from - a_to) % (2*math.pi))

ang_D_on_C1 = angle_of(Dxy - C1)
phi1 = ang_diff(ang_D_on_C1, ang_B, ccw=(sgn1 > 0))

ang_F_on_C2 = angle_of(np.array([Fx, Fy]) - C2)
phi2 = ang_diff(ang_F_on_C2, ang_D, ccw=(sgn2 > 0))

L1 = R1 * phi1
L2 = R2 * phi2


# ============= 螺线弧长原函数与反函数（只用于头把手的时间→参数） =============
def F_theta(theta: float) -> float:
    # F(θ) = 0.5*( θ*sqrt(1+θ^2) + asinh(θ) )
    return 0.5 * (theta*math.sqrt(1.0+theta*theta) + math.asinh(theta))

def inv_F(Fv: float, max_iter: int = 50, tol: float = 1e-13) -> float:
    theta = math.sqrt(max(0.0, 2.0*Fv)) if Fv>1.0 else max(0.0, Fv)
    for _ in range(max_iter):
        f = F_theta(theta) - Fv
        df = math.sqrt(1.0 + theta*theta)
        theta_new = theta - f/df
        if theta_new < 0.0: theta_new = 0.0
        if abs(theta_new - theta) < tol: return theta_new
        theta = theta_new
    return theta

# ============= 同段“等距螺线”两点弦长 = l 的 Δθ 求解（稳健） =============
def same_spiral_delta(theta1: float, l: float, *, offset: float = 0.0) -> float:
    """
    在 r = b*(θ - offset) 的同一条等距螺线上，已知第1点极角 θ1，
    求 Δ>0 使得两点间直线距离为 l。r1=b*(θ1-off), r2=b*(θ1+Δ-off)。
    """
    r1 = b * (theta1 - offset)
    if r1 < 1e-12: r1 = 1e-12

    def g(delta: float) -> float:
        r2 = b * (theta1 + delta - offset)
        return r1*r1 + r2*r2 - 2.0*r1*r2*math.cos(delta) - l*l

    def gp(delta: float) -> float:
        r2 = b * (theta1 + delta - offset)
        return 2.0*r2*b - 2.0*r1*b*math.cos(delta) + 2.0*r1*r2*math.sin(delta)

    delta = min(max(l/max(r1,1e-9), 1e-10), math.pi/2)  # 初值
    ok = False
    for _ in range(20):
        val = g(delta); der = gp(delta)
        if abs(der) < 1e-14: break
        cand = delta - val/der
        if not (0.0 < cand < math.pi): cand = 0.5*(delta + max(1e-10, min(cand, math.pi-1e-10)))
        delta = cand
        if abs(val) < 1e-12: ok = True; break
    if ok: return delta

    # 二分兜底
    lo, hi = 1e-12, math.pi - 1e-12
    gl, gh = g(lo), g(hi)
    if gl*gh > 0:
        hi = min(2*math.pi, hi + 1.0); gh = g(hi)
        if gl*gh > 0: return delta
    for _ in range(80):
        mid = 0.5*(lo+hi); gm = g(mid)
        if gm==0.0 or (hi-lo)<1e-12: return mid
        if gl*gm <= 0.0: hi, gh = mid, gm
        else:            lo, gl = mid, gm
    return 0.5*(lo+hi)

# ============= 段内坐标表达 =============
def xy_on_in(theta: float) -> np.ndarray:
    r = b * theta
    return np.array([r*math.cos(theta), r*math.sin(theta)], float)

def xy_on_out(theta: float) -> np.ndarray:
    r = b * (theta - math.pi)
    return np.array([r*math.cos(theta), r*math.sin(theta)], float)

def xy_on_arc1(phi: float) -> np.ndarray:
    ang = ang_B + sgn1 * phi
    return C1 + R1 * np.array([math.cos(ang), math.sin(ang)], float)

def xy_on_arc2(phi: float) -> np.ndarray:
    ang = ang_D + sgn2 * phi
    return C2 + R2 * np.array([math.cos(ang), math.sin(ang)], float)

# ============= 头把手的“参数（分段）随时间”的确定 =============
def head_state_at_time(t: float):
    """
    返回 (seg_id, param)：
      seg=1: param=θ_in（盘入）
      seg=2: param=φ1（大弧，从B起量）
      seg=3: param=φ2（小弧，从D起量）
      seg=4: param=θ_out（盘出）
    t=0 在触边点 B。
    """
    if t <= 0.0:
        # 在盘入螺线：从 B 往回走 |t|，F(θ)=F(θ_B)+|t|/b
        Fu = F_theta(theta_B) + (-t)/b
        theta = inv_F(Fu)
        return 1, theta
    if t <= L1:
        return 2, t / R1
    if t <= L1 + L2:
        return 3, (t - L1) / R2
    # 盘出螺线
    s_out = t - L1 - L2
    # u=θ-π：F(u) = F(θ_B) + s_out/b
    Fu = F_theta(theta_B) + s_out/b
    u = inv_F(Fu)
    theta = u + math.pi
    return 4, theta

# ============= “从当前把手回推下一把手”的递推（同截图思路） =============
def step_prev(seg: int, param: float, l: float) -> Tuple[int, float]:
    """
    已知当前把手所处段与段内参数，回推“上一个把手”的段与参数，使两点直线距离= l。
    逻辑：
      - 若在同一段：直接用解析式/牛顿（螺线）或 Δφ=2arcsin(l/2R)（圆）
      - 跨段：在当前段尽量消耗；若不够，再把剩余弦长交给上一段继续算
    上一段的顺序是： 2->1, 3->2, 4->3 。
    """
    if seg == 1:
        # 盘入螺线，同段无上界；直接求 Δθ
        dth = same_spiral_delta(param, l, offset=0.0)
        return 1, param + dth

    elif seg == 2:
        # 大圆弧（B→D）：若 φ >= Δφ 则同段；否则跨到盘入螺线
        if l >= 2.0*R1: l = 2.0*R1 - 1e-12
        dphi = 2.0 * math.asin(l/(2.0*R1))
        if param >= dphi + 1e-14:
            return 2, param - dphi
        # 跨段：先回到 B，消耗 chord_B = 2R1 sin(param/2)
        chord_used = 2.0 * R1 * math.sin(max(0.0, param)/2.0)
        l_rem = max(0.0, l - chord_used)
        # 上一段：盘入螺线，起点就是 B（θ=θ_B）
        if l_rem <= 1e-14:
            return 1, theta_B
        dth = same_spiral_delta(theta_B, l_rem, offset=0.0)
        return 1, theta_B + dth

    elif seg == 3:
        # 小圆弧（D→F）：若 φ >= Δφ 同段；否则跨到大圆弧
        if l >= 2.0*R2: l = 2.0*R2 - 1e-12
        dphi = 2.0 * math.asin(l/(2.0*R2))
        if param >= dphi + 1e-14:
            return 3, param - dphi
        # 跨段：先回到 D，消耗 chord_D = 2R2 sin(param/2)
        chord_used = 2.0 * R2 * math.sin(max(0.0, param)/2.0)
        l_rem = max(0.0, l - chord_used)
        if l_rem <= 1e-14:
            return 2, phi1  # 到达大弧末端（即 B→D 的 D 端），在大弧的参数就是 φ1
        # 进入大圆弧（反向，从 D 往 B 方向）：同理用 Δφ1
        dphi1 = 2.0 * math.asin(min(1.0, l_rem/(2.0*R1)))
        if phi1 >= dphi1 + 1e-14:
            return 2, phi1 - dphi1
        # 跨到盘入（再减一次）
        chord_used2 = 2.0 * R1 * math.sin(max(0.0, phi1)/2.0)
        l_rem2 = max(0.0, l_rem - chord_used2)
        if l_rem2 <= 1e-14:
            return 1, theta_B
        dth = same_spiral_delta(theta_B, l_rem2, offset=0.0)
        return 1, theta_B + dth

    else:  # seg == 4
        # 盘出螺线：同段 Δθ（注意 offset=π），若不够（理论上总是够），就跨到小圆
        dth = same_spiral_delta(param, l, offset=math.pi)
        # 检查是否跨越到 F（θ_F = θ_B + π）
        if param - dth >= theta_B + math.pi - 1e-12:
            return 4, param - dth
        # 跨段：先回到 F，消耗 chord_F
        P_now = xy_on_out(param)
        P_F = np.array([Fx, Fy], float)
        chord_to_F = float(np.hypot(*(P_now - P_F)))
        l_rem = max(0.0, l - chord_to_F)
        if l_rem <= 1e-14:
            return 3, phi2  # 到达小弧末端 F（在小弧的参数为 phi2）
        # 进入小圆（从 F 往 D 方向）：Δφ2
        dphi2 = 2.0 * math.asin(min(1.0, l_rem/(2.0*R2)))
        if phi2 >= dphi2 + 1e-14:
            return 3, phi2 - dphi2
        # 再跨到大圆、再跨到盘入，做法同上（这里省略到极端情形，通常不会一步跨三段）
        chord_used2 = 2.0 * R2 * math.sin(max(0.0, phi2)/2.0)
        l_rem2 = max(0.0, l - chord_to_F - chord_used2)
        if l_rem2 <= 1e-14:
            return 2, phi1
        dphi1 = 2.0 * math.asin(min(1.0, l_rem2/(2.0*R1)))
        if phi1 >= dphi1 + 1e-14:
            return 2, phi1 - dphi1
        chord_used3 = 2.0 * R1 * math.sin(max(0.0, phi1)/2.0)
        l_rem3 = max(0.0, l - chord_to_F - chord_used2 - chord_used3)
        if l_rem3 <= 1e-14:
            return 1, theta_B
        dth2 = same_spiral_delta(theta_B, l_rem3, offset=0.0)
        return 1, theta_B + dth2


# ============= Dragon 类（与截图相同结构） =============
class Dragon:
    def __init__(self, v0=1.0):
        self.v0 = v0      # 头把手速度（m/s）
        self.ang = None   # 各把手的“段内参数”数组（[(seg, param), ...]）
        self.pos = None   # 各把手坐标
        self.time = None  # 当前时刻
        self.vol = None   # 各把手速度（可选：本版默认不用解析式求，全局二阶差分更稳）

    # 给定 t，更新所有把手
    def set_time(self, t: float, need_vol=False):
        self.time = t

        # 头把手（段/参数）
        seg0, par0 = head_state_at_time(t)
        self.ang = [(seg0, par0)]
        # 头把手坐标
        P0 = (xy_on_in(par0) if seg0==1 else
              xy_on_arc1(par0) if seg0==2 else
              xy_on_arc2(par0) if seg0==3 else
              xy_on_out(par0))
        self.pos = [tuple(P0)]

        # 逐节回推：第1节距 2.86，之后 1.65，最后尾板再 1.65
        seg, par = seg0, par0
        # 龙头后=第1节前
        seg, par = step_prev(seg, par, l_HEAD)
        self.ang.append((seg, par))
        P = (xy_on_in(par) if seg==1 else
             xy_on_arc1(par) if seg==2 else
             xy_on_arc2(par) if seg==3 else
             xy_on_out(par))
        self.pos.append(tuple(P))

        # 第2..221节前把手
        for _ in range(2, 222):
            seg, par = step_prev(seg, par, l_BODY)
            self.ang.append((seg, par))
            P = (xy_on_in(par) if seg==1 else
                 xy_on_arc1(par) if seg==2 else
                 xy_on_arc2(par) if seg==3 else
                 xy_on_out(par))
            self.pos.append(tuple(P))

        # 龙尾前把手（再退 1.65）
        seg, par = step_prev(seg, par, l_BODY)
        self.ang.append((seg, par))
        P = (xy_on_in(par) if seg==1 else
             xy_on_arc1(par) if seg==2 else
             xy_on_arc2(par) if seg==3 else
             xy_on_out(par))
        self.pos.append(tuple(P))

        # 龙尾后把手（再退 1.65）
        seg, par = step_prev(seg, par, l_TAIL)
        self.ang.append((seg, par))
        P = (xy_on_in(par) if seg==1 else
             xy_on_arc1(par) if seg==2 else
             xy_on_arc2(par) if seg==3 else
             xy_on_out(par))
        self.pos.append(tuple(P))

        # 速度：如果需要，可在外部用小步长二阶差分对 Dragon.set_time(t±dt) 做数值速度。
        if need_vol:
            self.vol = None  # 占位——建议在外部用数值法求

    # 简单画图检查（螺线片段 + 两个圆弧 + 全队把手点）
    def print_status(self):
        # 画调头圆
        fig, ax = plt.subplots(figsize=(6,6))
        circ = plt.Circle((0,0), r_turn, color='yellow', alpha=0.2)
        ax.add_artist(circ)

        # 盘入/盘出螺线（只画一小段示意）
        thetas = np.linspace(theta_B, theta_B+4.0, 600)  # 盘入外侧示意
        xs = b*thetas*np.cos(thetas); ys = b*thetas*np.sin(thetas)
        ax.plot(xs, ys, 'r-', lw=1.5, alpha=0.5, label='盘入螺线(示意)')

        thetas2 = np.linspace(theta_B+math.pi, theta_B+math.pi+4.0, 600)
        rs2 = b*(thetas2 - math.pi)
        ax.plot(rs2*np.cos(thetas2), rs2*np.sin(thetas2),
                'b-', lw=1.5, alpha=0.5, label='盘出螺线(示意)')

        phis1 = np.linspace(0.0, phi1, 200)
        P1 = C1 + R1 * np.c_[np.cos(ang_B + sgn1 * phis1), np.sin(ang_B + sgn1 * phis1)]
        ax.plot(P1[:, 0], P1[:, 1], 'k-', lw=2, alpha=0.8)

        phis2 = np.linspace(0.0, phi2, 200)
        P2 = C2 + R2 * np.c_[np.cos(ang_D + sgn2 * phis2), np.sin(ang_D + sgn2 * phis2)]
        ax.plot(P2[:, 0], P2[:, 1], 'k-', lw=2, alpha=0.8)

        # 标注 B、D、F、圆心
        ax.scatter([Bx, Dxy[0], Fx, O1x, O2x],
                   [By, Dxy[1], Fy, O1y, O2y],
                   c=['crimson','green','crimson','black','black'],
                   s=[50,50,50,30,30], zorder=5)
        ax.annotate("B", (Bx,By), xytext=(5,5), textcoords='offset points')
        ax.annotate("D", (Dxy[0],Dxy[1]), xytext=(5,5), textcoords='offset points')
        ax.annotate("F", (Fx,Fy), xytext=(5,5), textcoords='offset points')

        # 把手散点
        xs = [p[0] for p in self.pos]; ys = [p[1] for p in self.pos]
        ax.scatter(xs, ys, c='k', s=10, zorder=6)

        ax.set_aspect('equal', 'box')
        ax.grid(True, ls='--', alpha=0.3)
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()


# ============= 示例运行 =============
if __name__ == "__main__":
    print(f"[Geom] θ_B={theta_B:.9f}, α={alpha:.9f}")
    print(f"[Geom] C1=({O1x:.6f},{O1y:.6f}), R1={R1:.6f}, φ1={phi1:.9f}, L1={L1:.6f}")
    print(f"[Geom] C2=({O2x:.6f},{O2y:.6f}), R2={R2:.6f}, φ2={phi2:.9f}, L2={L2:.6f}")
    print(f"[Geom] B=({Bx:.6f},{By:.6f}), D=({Dxy[0]:.6f},{Dxy[1]:.6f}), F=({Fx:.6f},{Fy:.6f})")

    dragon = Dragon(v0=1.0)

    # 例：t=0 在 B；t=50s 在大/小弧或盘出；t=-50s 在盘入
    for t in (-50, 0, 50):
        dragon.set_time(t)
        head = dragon.pos[0]
        print(f"[t={t:>4}] head=({head[0]:.6f}, {head[1]:.6f})")
    # 画一张示意（最后一次 set_time 的状态）
    dragon.print_status()