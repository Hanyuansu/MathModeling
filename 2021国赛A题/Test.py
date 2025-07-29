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
