import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Problem1 import boundary,environment_temperature,segment_index,simulate_airzone_piecewise,objective
from Problem2 import check
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# === 数据读取与预处理 ===
df = pd.read_excel("附件.xlsx")
df.columns = ['time', 'temperature']
df.dropna(inplace=True)
t = df['time'].values
T_data = df['temperature'].values
dt = np.diff(t)[0]
v_default = 70 / 60
x = v_default * t

# === 几何参数 ===
d = 0.15e-3
dx = d / 2

def compute_area(t_sim, T_sim):
    idx_217 = np.where(T_sim >= 217)[0]
    if len(idx_217) < 2:
        return np.inf
    t3 = idx_217[0]
    tm = np.argmax(T_sim)
    if not (t3 < tm):
        return np.inf
    dt = t_sim[1] - t_sim[0]
    integrand = T_sim[t3:tm + 1] - 217
    return np.trapz(integrand, dx=dt)

def random_search_area_minimize(best_params, N=10000):
    T_base = np.array([175, 195, 235, 255])
    v_base = 65
    s = 0
    i = 1
    minS = []
    best_result = None

    while s < N:
        x = np.random.rand(5)
        T_set0 = T_base + (x[:4] - 0.5) * 20
        v0 = v_base + 35 * x[4]

        t_sim = np.arange(0, 700, 0.5)
        x_sim = v0 / 60 * t_sim
        T0_sim = np.array([environment_temperature(T_set0, xi) for xi in x_sim])
        T_sim = simulate_airzone_piecewise(best_params, t_sim, x_sim, T0_sim)

        if not check(t_sim, T_sim):
            S = np.inf
        else:
            S = compute_area(t_sim, T_sim)
            if S == 0:
                S = np.inf

        if i == 1 or S < min(minS):
            best_result = (T_set0, v0, S, t_sim, T_sim)
            s = 0
        else:
            s += 1

        minS.append(S)
        i += 1

    return best_result, minS
# === 绘制温度区间图 ===
def plot_temperature_profile(T_set, v, params):
    t_sim = np.arange(0, 500, 0.5)
    x_sim = v / 60 * t_sim
    T0_profile = np.array([environment_temperature(T_set, xi) for xi in x_sim])
    T_sim = simulate_airzone_piecewise(params, t_sim, x_sim, T0_profile)

    plt.figure(figsize=(10, 5))
    plt.plot(t_sim, T0_profile, label='环境温度', color='blue')
    plt.plot(t_sim, T_sim, label='芯片中心温度', color='red')
    plt.xlabel('时间/s')
    plt.ylabel('温度/℃')
    plt.title('环境温度与芯片中心温度曲线')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

#实际分区温度
T_set = [175, 195, 235, 255]
T0 = np.array([environment_temperature(T_set, xi) for xi in x])

#初始化和边界
n_segments = 11
initial_guess = [1e-6, 30] * n_segments
bounds = [(1e-8, 1e-4), (1, 200)] * n_segments
result = minimize(objective, initial_guess, bounds=bounds)
best_params = result.x
# 示例调用（你可替换为实际最优解）

(best_T_set, best_v, best_S, t_best, T_best), minS_history = random_search_area_minimize(best_params)
print("最优速度 v =", best_v, "cm/min")
print("最优温度设置:", best_T_set)
print("最小面积目标值 =", best_S)

plot_temperature_profile(best_T_set,best_v,best_params)

plt.figure()
plt.plot(range(len(minS_history)), minS_history, 'k')
plt.xlabel('迭代次数')
plt.ylabel('面积值')
plt.title('随机搜索面积收敛曲线')
plt.grid(True)
plt.tight_layout()
plt.show()
