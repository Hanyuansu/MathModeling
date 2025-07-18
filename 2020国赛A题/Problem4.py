import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Problem1 import boundary, environment_temperature, segment_index, simulate_airzone_piecewise, objective
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

# === 面积计算 ===
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
    return np.trapezoid(integrand, dx=dt)

# === 对称性指标计算 ===
def compute_symmetry(t_sim, T_sim):
    idx_217 = np.where(T_sim >= 217)[0]
    if len(idx_217) < 2:
        return np.inf
    t3 = idx_217[0]
    t4 = idx_217[-1]
    tm = np.argmax(T_sim)
    if tm <= t3 or tm >= t4:
        return np.inf
    dt = t_sim[1] - t_sim[0]
    left = np.trapezoid(T_sim[t3:tm], dx=dt)
    right = np.trapezoid(T_sim[tm:t4], dx=dt)
    return (left - right) ** 2

# === 动态组合目标函数生成器 ===
def make_combined_objective(area_min, sym_min):
    def _combined(area, sym):
        return area / area_min + sym / sym_min
    return _combined

# === 随机搜索组合优化 ===
def random_search_multiobjective(best_params, N=10000, combined_func=None):
    T_base = np.array([175, 195, 235, 255])
    v_base = 65
    s = 0
    i = 1
    best_result = None
    min_obj = np.inf
    history = []

    while s < N:
        x = np.random.rand(5)
        T_set0 = T_base + (x[:4] - 0.5) * 20
        v0 = v_base + 35 * x[4]

        t_sim = np.arange(0, 700, 0.5)
        x_sim = v0 / 60 * t_sim
        T0_sim = np.array([environment_temperature(T_set0, xi) for xi in x_sim])
        T_sim = simulate_airzone_piecewise(best_params, t_sim, x_sim, T0_sim)

        area = compute_area(t_sim, T_sim)
        sym = compute_symmetry(t_sim, T_sim)

        if not check(t_sim, T_sim) or area == np.inf or sym == np.inf or area < area_min or sym < sym_min:
            obj = 1e6
        else:
            obj = combined_func(area, sym)

        if obj < min_obj:
            best_result = (T_set0, v0, area, sym, t_sim, T_sim, obj)
            min_obj = obj
            s = 0
        else:
            s += 1

        history.append(obj)
        i += 1

    return best_result, history

# === 单目标优化：面积 ===
def optimize_single_objective_area(params, N=3000):
    best_result = None
    min_area = np.inf
    for _ in range(N):
        x = np.random.rand(5)
        T_set0 = np.array([175, 195, 235, 255]) + (x[:4] - 0.5) * 20
        v0 = 65 + 35 * x[4]

        t_sim = np.arange(0, 700, 0.5)
        x_sim = v0 / 60 * t_sim
        T0_sim = np.array([environment_temperature(T_set0, xi) for xi in x_sim])
        T_sim = simulate_airzone_piecewise(params, t_sim, x_sim, T0_sim)

        area = compute_area(t_sim, T_sim)

        if check(t_sim, T_sim) and area != np.inf and area < min_area:
            best_result = (T_set0, v0, area)
            min_area = area

    return min_area if best_result else 1e3

# === 单目标优化：对称性 ===
def optimize_single_objective_sym(params, N=3000):
    best_result = None
    min_sym = np.inf
    for _ in range(N):
        x = np.random.rand(5)
        T_set0 = np.array([175, 195, 235, 255]) + (x[:4] - 0.5) * 20
        v0 = 65 + 35 * x[4]

        t_sim = np.arange(0, 700, 0.5)
        x_sim = v0 / 60 * t_sim
        T0_sim = np.array([environment_temperature(T_set0, xi) for xi in x_sim])
        T_sim = simulate_airzone_piecewise(params, t_sim, x_sim, T0_sim)

        sym = compute_symmetry(t_sim, T_sim)

        if check(t_sim, T_sim) and sym != np.inf and sym < min_sym:
            best_result = (T_set0, v0, sym)
            min_sym = sym

    return min_sym if best_result else 1e3

# === 绘图函数 ===
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

# === 主程序 ===
if __name__ == "__main__":
    T_set = [175, 195, 235, 255]
    T0 = np.array([environment_temperature(T_set, xi) for xi in x])
    initial_guess = [1e-6, 30] * 11
    bounds = [(1e-8, 1e-4), (1, 200)] * 11

    print("正在拟合最优传热参数...")
    result = minimize(objective, initial_guess, bounds=bounds)
    best_params = result.x

    print("正在搜索面积最小值...")
    area_min = optimize_single_objective_area(best_params)
    print("正在搜索对称性最小值...")
    sym_min = optimize_single_objective_sym(best_params)
    print("归一化参数：面积最小值 =", round(area_min, 2), ", 对称性最小值 =", round(sym_min, 2))

    combined_func = make_combined_objective(area_min, sym_min)
    print("正在执行多目标组合优化搜索...")
    (best_T_set, best_v, area_val, sym_val, t_best, T_best, final_obj), history = random_search_multiobjective(best_params, combined_func=combined_func)

    print("\n=== 最终优化结果 ===")
    print("最优速度 v =", round(best_v), "cm/min")
    print("最优温度设置:", np.round(best_T_set).astype(int))
    print("最小面积 =", round(area_val, 2))
    print("对称性指标 =", round(sym_val, 2))
    print("最终目标函数值 =", round(final_obj, 4))

    plot_temperature_profile(best_T_set, best_v, best_params)

    plt.figure()
    plt.plot(history, 'k')
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数值')
    plt.title('组合目标函数收敛曲线')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

