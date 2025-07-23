import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# === 1. 读取数据 ===
df = pd.read_excel("附件1.xlsx")
x = df['X坐标（米）'].values
y = df['Y坐标（米）'].values
z = df['Z坐标（米）'].values

# === 2. 计算球面半径 ===
r_data = np.sqrt(x**2 + y**2 + z**2)
R = np.mean(r_data)

# === 3. 参数设置 ===
F = 0.466 * R
r = 150.0
stroke_lim = 0.6

# 提取 xz 平面上的点
x_proj = np.linspace(-150, 150, 100000)

# === 4. 计算误差函数 ===
def compute_errors(delta):
    zp = -1*R - delta + x_proj**2 / (4 * (F + delta))  # 抛物面拟合
    norm = np.sqrt(x_proj**2 + zp**2)
    xb = x_proj / norm * R
    zb = zp / norm * R
    errors = np.sqrt((x_proj - xb)**2 + (zp - zb)**2)
    return errors

# === 5. 遍历 δ 并记录最大误差 ===
delta_vals = np.arange(-0.6, 0.6, 0.001)
max_errors = []
mean_errors = []

for delta in delta_vals:
    errs = compute_errors(delta)
    max_errors.append(np.max(errs))
    mean_errors.append(np.mean(errs))

# 找出最小最大误差对应的 δ
delta_min_mean = delta_vals[np.argmin(mean_errors)]
min_mean_error = np.min(mean_errors)

delta_min_max = delta_vals[np.argmin(max_errors)]
min_max_error = np.min(max_errors)

# === 6. 绘图 ===
plt.figure(figsize=(10, 6))
plt.plot(delta_vals, max_errors, label='最大误差', color='orangered')
plt.plot(delta_vals, mean_errors, label='平均误差', color='cornflowerblue')
plt.axvline(x=delta_min_max, linestyle='--', color='gray', label=f'最优 δ ≈ {delta_min_max:.2f} m')
plt.axvline(x=delta_min_mean, linestyle='--', color='gray', label=f'最优 δ ≈ {delta_min_mean:.2f} m')
plt.xlabel('δ（抛物面顶点上移量，单位：m）')
plt.ylabel('径向误差（单位：m）')
plt.title('δ 与 最大/平均径向误差 的关系图')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === 7. 输出最优结果 ===
print(f"最优 δ = {delta_min_mean:.3f} m，对应最大误差 = {min_mean_error:.4f} m")
# z=-R-δ+x^2/4(F+δ),F=0.466R
# x^2+y^2=561.5*(z+300.79)