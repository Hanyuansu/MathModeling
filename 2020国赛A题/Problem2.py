import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Problem1 import boundary,environment_temperature,segment_index,simulate_airzone_piecewise,objective
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel("附件.xlsx")
df.columns = ['time', 'temperature']
df.dropna(inplace=True)
t = df['time'].values
T_data = df['temperature'].values
dt = np.diff(t)[0]
v = 70 / 60
x = v * t

d = 0.15e-3
dx = d / 2

T_set = [175, 195, 235, 255]
T0 = np.array([environment_temperature(T_set, xi) for xi in x])

n_segments = 11
initial_guess = [1e-6, 30] * n_segments
bounds = [(1e-8, 1e-4), (1, 200)] * n_segments

result = minimize(objective, initial_guess, bounds=bounds)
best_params = result.x

#制程界限
def check(t, T):
    dt = np.diff(t)
    dT = np.diff(T)
    dT_dt = dT / dt

    # （16）升温 / 降温斜率限制
    if not np.all((dT_dt >= -3) & (dT_dt <= 3)):
        return False

    # （17）温度上升区间（150~190°C）
    idx_150 = np.where(T >= 150)[0]
    idx_190 = np.where(T >= 190)[0]
    if len(idx_150) == 0 or len(idx_190) == 0:
        return False
    t1 = t[idx_150[0]]
    t2 = t[idx_190[0]]
    if not (60 <= t2 - t1 <= 120):
        return False
    # 且中间持续上升
    if not np.all(np.diff(T[idx_150[0]:idx_190[0] + 1]) >= 0):
        return False

    # （18）超过217°C的时间限制
    idx_217 = np.where(T >= 217)[0]
    if len(idx_217) == 0:
        return False
    t3 = t[idx_217[0]]
    t4 = t[idx_217[-1]]
    if not (40 <= t4 - t3 <= 90):
        return False

    # （19）峰值温度限制
    Tmax = np.max(T)
    if not (240 <= Tmax <= 250):
        return False
    return True

#二分搜索
def binary_search_max_speed(params, T_set, v_low=65, v_high=100, tol=0.01):
    best_v = -1
    while v_high - v_low > tol:
        v_mid = (v_low + v_high) / 2
        t_sim = np.arange(0, 500, 0.5)
        x_sim = v_mid / 60 * t_sim  # 单位换算成 cm/s
        T0_sim = np.array([environment_temperature(T_set, xi) for xi in x_sim])
        T_sim = simulate_airzone_piecewise(params, t_sim, x_sim, T0_sim)

        if check(t_sim, T_sim):
            best_v = v_mid
            v_low = v_mid  # 尝试更大的速度
        else:
            v_high = v_mid  # 降低速度
    return best_v
T_set=[182,203,237,254]
v_max = binary_search_max_speed(best_params, T_set)
print(f"最大满足工艺约束的过炉速度：{v_max:.2f} cm/min")
