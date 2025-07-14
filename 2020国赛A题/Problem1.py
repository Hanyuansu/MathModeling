import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

#读数据
df = pd.read_excel("附件.xlsx")
df.columns = ['time', 'temperature']
df.dropna(inplace=True)
t = df['time'].values
T_data = df['temperature'].values
dt = np.diff(t)[0]
v = 70 / 60
x = v * t

#几何参数
d = 0.15e-3
dx = d / 2#空间上离散成三个点

#分段函数(domain)
def boundary():
    D, d, D0 = 30.5, 5, 25
    X = np.zeros(11)
    X[0] = 0
    X[1] = D0
    X[2] = X[1] + 5 * D + 4 * d
    X[3] = X[2] + d
    X[4] = X[3] + D
    X[5] = X[4] + d
    X[6] = X[5] + D
    X[7] = X[6] + d
    X[8] = X[7] + 2 * D + d
    X[9] = X[8] + 2 * (d + D)
    X[10] = X[9] + D0
    return X

#炉温温度分段函数
def environment_temperature(T_set, x):
    X = boundary()
    if 0 <= x < 20:
        return 25
    elif 20 <= x <= X[1]:
        return (T_set[0] - 25) / (X[1] - 20) * (x - 20) + 25
    elif X[1] < x <= X[2]:
        return T_set[0]
    elif X[2] < x <= X[3]:
        return (T_set[1] - T_set[0]) / (X[3] - X[2]) * (x - X[2]) + T_set[0]
    elif X[3] < x <= X[4]:
        return T_set[1]
    elif X[4] < x <= X[5]:
        return (T_set[2] - T_set[1]) / (X[5] - X[4]) * (x - X[4]) + T_set[1]
    elif X[5] < x <= X[6]:
        return T_set[2]
    elif X[6] < x <= X[7]:
        return (T_set[3] - T_set[2]) / (X[7] - X[6]) * (x - X[6]) + T_set[2]
    elif X[7] < x <= X[8]:
        return T_set[3]
    elif X[8] < x <= X[9]:
        return (25 - T_set[3]) / (X[9] - X[8]) * (x - X[8]) + T_set[3]
    elif X[9] < x <= X[10]:
        return 25
    else:
        return 25

#判断当前位置所在分区
def segment_index(xi):
    X = boundary()
    if xi < 20:
        return 0
    elif xi <= X[1]:
        return 1
    elif xi <= X[2]:
        return 2
    elif xi <= X[3]:
        return 3
    elif xi <= X[4]:
        return 4
    elif xi <= X[5]:
        return 5
    elif xi <= X[6]:
        return 6
    elif xi <= X[7]:
        return 7
    elif xi <= X[8]:
        return 8
    elif xi <= X[9]:
        return 9
    elif xi <= X[10]:
        return 10
    else:
        return 10

#PDE数值求解，显示形式，每个分区求解2个参数
def simulate_airzone_piecewise(params, t, x, T0):
    u = np.zeros((3, len(t)))
    u[:, 0] = 25
    T_center = np.zeros(len(t))
    T_center[0] = 25
    for j in range(len(t) - 1):
        seg = segment_index(x[j])
        a = params[2 * seg]
        b = params[2 * seg + 1]
        r = a ** 2 * dt / dx ** 2
        u[1, j + 1] = r * u[0, j] + (1 - 2 * r) * u[1, j] + r * u[2, j]
        u[0, j + 1] = (u[1, j + 1] + b * dx * T0[j + 1]) / (1 + b * dx)
        u[2, j + 1] = u[0, j + 1]
        T_center[j + 1] = u[1, j + 1]
    return T_center

#目标函数
def objective(params):
    T_pred = simulate_airzone_piecewise(params, t, x, T0)
    return np.sqrt(np.mean((T_pred - T_data) ** 2))

#实际分区温度
T_set = [175, 195, 235, 255]
T0 = np.array([environment_temperature(T_set, xi) for xi in x])

#初始化和边界
n_segments = 11
initial_guess = [1e-6, 30] * n_segments
bounds = [(1e-8, 1e-4), (1, 200)] * n_segments
result = minimize(objective, initial_guess, bounds=bounds)
best_params = result.x
T_fit = simulate_airzone_piecewise(best_params, t, x, T0)

#输出
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(t, T_data, 'r.', label='真实数据')
plt.plot(t, T_fit, 'b-', label='拟合中心温度')
plt.xlabel("时间/s")
plt.ylabel("温度/°C")
plt.title("温度拟合曲线")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
residual = T_fit - T_data
rmse = np.sqrt(np.mean(residual ** 2))
plt.plot(t, residual, 'k')
plt.xlabel("时间/s")
plt.ylabel("残差")
plt.title(f"残差图（RMSE={rmse:.2f} °C）")
plt.grid(True)

plt.tight_layout()
plt.show()

#炉温曲线
plt.figure(figsize=(8, 5))
plt.plot(t, T_fit, label='焊接区域中心温度', color='blue')
plt.xlabel("时间 / s")
plt.ylabel("温度 / °C")
plt.title("焊接区域中心温度随时间变化曲线")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("焊接区域中心温度曲线.png", dpi=300)
plt.show()

X=boundary()
v = 78 / 60
# 计算小温区中点位置与结束位置所对应的时间
pos_z3 = (X[1] + X[2]) / 2
pos_z6 = (X[4] + X[5]) / 2
pos_z7 = (X[6] + X[7]) / 2
pos_z8 = X[8]

# 对应位置所处时间
t_z3 = pos_z3 / v
t_z6 = pos_z6 / v
t_z7 = pos_z7 / v
t_z8 = pos_z8 / v

# 插值查温度
T_z3 = np.interp(t_z3, t, T_fit)
T_z6 = np.interp(t_z6, t, T_fit)
T_z7 = np.interp(t_z7, t, T_fit)
T_z8 = np.interp(t_z8, t, T_fit)

print(f"小温区 3 中点中心温度: {T_z3:.2f} °C")
print(f"小温区 6 中点中心温度: {T_z6:.2f} °C")
print(f"小温区 7 中点中心温度: {T_z7:.2f} °C")
print(f"小温区 8 结束处中心温度: {T_z8:.2f} °C")


save_times = np.arange(0, t[-1], 0.5)  # 每0.5秒的时间点
save_positions = v * save_times  # 对应位置（使用题设速度）
save_temperatures = np.interp(save_positions, x, T_fit)

# 保存为 result.csv
df_result = pd.DataFrame({
    'time(s)': save_times,
    'center_temperature(°C)': save_temperatures
})
df_result.to_csv("result.csv", index=False, encoding='utf-8-sig')

#隐式形式
# def simulate_airzone_piecewise(params, t, x, T0):
#     """
#     Crank–Nicolson 隐式二阶格式 + Robin 边界解析更新
#     时间二阶、空间二阶精度，不用显式解矩阵，只用一行公式更新中心点。
#     params: 长度 2*n_segments，偶数位 a，奇数位 b
#     t, x: 时间和对应位置
#     T0: 环境温度随时间的数组
#     """
#     n = len(t)
#     u0 = np.zeros(n)      # 左表面
#     u1 = np.zeros(n)      # 中心
#     u2 = np.zeros(n)      # 右表面
#     u0[0] = u1[0] = u2[0] = 25.0
#
#     for j in range(n - 1):
#         seg = segment_index(x[j])
#         a = params[2*seg]
#         b = params[2*seg + 1]
#
#         # 无量纲数
#         r = a**2 * dt / dx**2
#         # Robin 边界系数
#         alpha = 1.0 / (1 + b*dx)      # u0 = alpha*u1 + (1-alpha)*T0
#         beta  = b*dx * alpha
#
#         # —— Crank–Nicolson 更新中心温度 ——
#         # u1^{n+1} = [ (1 - r + r*alpha)*u1^n  +  r*beta*(T0^n + T0^{n+1}) ]
#         #             -----------------------------------------------
#         #                        1 + r*beta
#         num = (1 - r + r*alpha) * u1[j] + r*beta * (T0[j] + T0[j+1])
#         den = 1 + r*beta
#         u1[j+1] = num / den
#
#         # —— Robin 边界（左右对称）——
#         u0[j+1] = alpha * u1[j+1] + beta * T0[j+1]
#         u2[j+1] = u0[j+1]
#
#     return u1
# def simulate_airzone_piecewise(params, t, x, T0):
#     """
#     基于后向欧拉隐式格式求解一维热传导＋对流（Robin）边界条件
#     保持时间一阶、空间二阶精度，且无条件稳定。
#     params: 长度为2*n_segments的参数向量，偶数位是a，奇数位是b
#     t, x: 时间和对应位置
#     T0: 环境温度随时间的向量
#     """
#     u = np.zeros((3, len(t)))    # u[0]=左边界, u[1]=中心, u[2]=右边界
#     u[:, 0] = 25                 # 初始均为环境温
#     T_center = np.zeros(len(t))
#     T_center[0] = 25
#
#     for j in range(len(t) - 1):
#         seg = segment_index(x[j])
#         a = params[2*seg]        # 热扩散系数
#         b = params[2*seg + 1]    # 对流换热系数
#         # r = a^2 * Δt / Δx^2
#         r = a**2 * dt / dx**2
#
#         # ===== 隐式后向欧拉离散并结合对流边界，推导得到中心节点更新公式 =====
#         # (1) u1^{j+1} 项系数:      1 + b·dx + 2·r·b·dx
#         # (2) 右端项: (1 + b·dx)·u1^j + 2·r·b·dx·T0^{j+1}
#         numerator   = (1 + b*dx) * u[1, j] + 2 * r * b * dx * T0[j+1]
#         denominator = 1 + b*dx + 2 * r * b * dx
#         u1_new = numerator / denominator
#
#         # 更新对流边界：u0^{j+1}=u2^{j+1}=(u1^{j+1} + b·dx·T0^{j+1})/(1 + b·dx)
#         u0_new = (u1_new + b*dx * T0[j+1]) / (1 + b*dx)
#
#         # 存储
#         u[:, j+1] = [u0_new, u1_new, u0_new]
#         T_center[j+1] = u1_new
#
#     return T_center