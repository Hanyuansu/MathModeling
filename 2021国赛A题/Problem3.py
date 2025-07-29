import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

R = 300.4           # 球面半径
r0 = 150            # 接收面口径半径
delta = 0.5         # 接收面 ±0.5 米容差
theta_deg = np.linspace(0.01, 30, 10000)
theta_rad = np.radians(theta_deg)

# 计算 x_E 与 r(θ)
x_E = (R / (2 * np.cos(theta_rad)) - (1 - 0.466) * R) * np.tan(2 * theta_rad)
r = R * np.sin(theta_rad)

# 找出满足 |x_E| <= 0.5 的解
valid_mask = np.abs(x_E) <= delta
valid_indices = np.where(valid_mask)[0]

# 自动识别分段（两个非连续区间）
split_idx = np.where(np.diff(valid_indices) > 1)[0]


# 分离为两个索引区间
idx1 = valid_indices[:split_idx[0]+1]   # 小圆
idx2 = valid_indices[split_idx[0]+1:]   # 圆环
r1 = np.max(r[idx1])           # 小圆区域的最大极径
r2 = np.min(r[idx2])           # 圆环最小
r3 = np.max(r[idx2])           # 圆环最大

#面积与接收比计算
Sp = np.pi * r1**2 + np.pi * (r3**2 - r2**2)
eta_basic = Sp / (np.pi * r0**2) * 100

print(f"有效角度范围 θ: {theta_deg[idx1[0]]:.4f}° -- {theta_deg[idx2[-1]]:.4f}°")
print(f"r1 = {r1:.4f} m, r2 = {r2:.4f} m, r3 = {r3:.4f} m")
print(f"接收面积 Sp = {Sp:.4f} m²")
print(f"基准球面接收比 η_basic = {eta_basic:.4f}%")

plt.figure(figsize=(14, 6))

# --- 图 1: x_E vs θ
plt.subplot(1, 2, 1)
plt.plot(theta_deg, x_E, label='$x_E$', color='blue')
plt.axhline(y=0.5, color='red', linestyle='--', label=r'$x_E = \pm 0.5$ m')
plt.axhline(y=-0.5, color='red', linestyle='--')
plt.fill_between(theta_deg[idx1], x_E[idx1], color='blue', alpha=0.3, label='小圆接收区')
plt.fill_between(theta_deg[idx2], x_E[idx2], color='orange', alpha=0.3, label='圆环接收区')
plt.xlabel('入射角 θ (°)')
plt.ylabel('横向偏移 $x_E$ (m)')
plt.title('投影偏移 $x_E$ 与 θ 的关系')
plt.grid(True)
plt.legend()

# --- 图 2: x_E vs r ---
plt.subplot(1, 2, 2)
plt.plot(r, x_E, label='$x_E$ vs $r$', color='green')
plt.axhline(y=0.5, color='red', linestyle='--', label=r'$x_E = \pm 0.5$ m')
plt.axhline(y=-0.5, color='red', linestyle='--')
# 填充两个有效区间
plt.fill_between(r[idx1], x_E[idx1], color='blue', alpha=0.3, label='小圆接收区')
plt.fill_between(r[idx2], x_E[idx2], color='orange', alpha=0.3, label='圆环接收区')
plt.xlabel('极径 $r$ (m)')
plt.ylabel('投影偏移 $x_E$ (m)')
plt.title('投影偏移 $x_E$ 与极径 $r$ 的关系')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()




# === 参数定义 ===
R = 300.4                               # 球面半径
f_ratio = 0.466
f = f_ratio * R + 0.39                  # 焦距
h = 0.534 * R                           # 馈源舱高度
CP_unit = np.array([0,0,-1])  # 单位方向向量（从 C 指向 P）              # |CP|
N = 100                                 # 每个面片采样点数
accept_radius_squared = 0.5 ** 2

a=np.radians(36.795)
b=np.radians(90-78.169)
Rz = np.array([
    [ math.cos(a), math.sin(a), 0,],
    [-math.sin(a), math.cos(a), 0,],
    [ 0          , 0          , 1,]
])
Ry = np.array([
    [ math.cos(b), 0, -math.sin(b),],
    [ 0,           1,            0,],
    [ math.sin(b), 0,  math.cos(b),],
])
R0=Ry@Rz
CP_unit=R0@CP_unit.T
P = CP_unit * h

# === 数据读取 ===
df_nodes = pd.read_excel("应调整主索节点列表.xlsx")
df_reflector = pd.read_excel("附件3.xlsx", usecols=["主索节点1", "主索节点2", "主索节点3"])

# === 构建主索节点坐标字典 ===
coord_dict = {}
for _, row in df_nodes.iterrows():
    node_id = row["主索节点编号"]
    coord = np.array([row["X坐标（米）"], row["Y坐标（米）"], row["Z坐标（米）"]])
    coord_dict[node_id] = coord

df_faces = df_reflector[
    df_reflector["主索节点1"].isin(coord_dict) &
    df_reflector["主索节点2"].isin(coord_dict) &
    df_reflector["主索节点3"].isin(coord_dict)
].reset_index(drop=True)

# === 反解球心 ===
def solve_sphere_center(P1, P2, P3, R):
    def residuals(C):
        return [np.linalg.norm(C - P1) - R,
                np.linalg.norm(C - P2) - R,
                np.linalg.norm(C - P3) - R]
    C0 = np.array([0.0, 0.0, 0.0])
    result1 = least_squares(residuals, C0)
    C1 = result1.x
    mirror = C0 + 2 * (C0 - C1)
    result2 = least_squares(residuals, mirror)
    C2 = result2.x
    return C1 if np.linalg.norm(C1) < np.linalg.norm(C2) else C2

#点在平面投影三角形内判断（重心坐标法）
def is_inside_triangle(p, a, b, c):
    v0, v1, v2 = c - a, b - a, p - a
    dot00, dot01, dot02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
    dot11, dot12 = np.dot(v1, v1), np.dot(v1, v2)
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return (u >= 0) and (v >= 0) and (u + v <= 1)

total_Sw = 0
weighted_sum = 0
eta_list = []

for _, row in tqdm(df_faces.iterrows(), total=len(df_faces)):
    id1, id2, id3 = row["主索节点1"], row["主索节点2"], row["主索节点3"]
    A, B, C = coord_dict[id1], coord_dict[id2], coord_dict[id3]

    # 反解球心
    center = solve_sphere_center(A, B, C, R)

    # 入射方向 u
    u = np.array([0, 0, -1])

    # CP向量 = P - center
    CP = P - center
    CU = -u
    cos_theta = np.dot(CP, CU) / (np.linalg.norm(CP) * np.linalg.norm(CU))
    CF_len = R / (2 * cos_theta)
    F = center + CF_len * u
    reflect_dir = F - P
    reflect_dir /= np.linalg.norm(reflect_dir)

    # 反射线参数形式 UF: x = lt + x0...
    # 跟入射方向垂直的平面方程: xp*x + yp*y + zp*z = |CP|^2
    xp, yp, zp = CP
    RHS = np.dot(CP, CP)

    count_in = 0
    for _ in range(N):
        # t = np.random.rand(3)
        # t /= np.sum(t)
        # M = t[0] * A + t[1] * B + t[2] * C  # 三角形内采样点
        t = np.random.rand(3)
        t /= np.sum(t)
        M_flat = t[0] * A + t[1] * B + t[2] * C
        vec = M_flat - center
        vec /= np.linalg.norm(vec)
        M = center + R * vec  # 球面上点

        # 反射方向 = F - M
        D = F - M
        D /= np.linalg.norm(D)

        # 求交点 Q = M + λD
        numerator = RHS - np.dot([xp, yp, zp], M)
        denominator = np.dot([xp, yp, zp], D)
        if np.abs(denominator) < 1e-6:
            continue
        lam = numerator / denominator
        Q = M + lam * D

        # 判断是否落在 ABC 投影内（三点在同一平面）
        if is_inside_triangle(Q, A, B, C):
            count_in += 1

    Swp = count_in / N
    eta = Swp
    total_Sw += Swp
    eta_list.append(eta)

# === 最终接收比 ===
eta_array = np.array(eta_list)
w = eta_array / np.sum(eta_array)
total_eta = np.sum(w * eta_array)
print("总接收比 η ≈ {:.6f}".format(total_eta))

