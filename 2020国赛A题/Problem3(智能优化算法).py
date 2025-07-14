from deap import base, creator, tools, algorithms
import random
import numpy as np
import matplotlib.pyplot as plt
from Problem1 import boundary, environment_temperature, segment_index, simulate_airzone_piecewise, objective
from Problem2 import check
from scipy.optimize import minimize

# === 面积函数 ===
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

# === 创建遗传算法适应度函数（面积最小化） ===
creator.create("FitnessMin_Area", base.Fitness, weights=(-1.0,))
creator.create("Individual_Area", list, fitness=creator.FitnessMin_Area)

def evaluate_area(individual, best_params):
    # 限定 ±10°C 浮动并加入 clip，避免突破边界
    T1 = np.clip(175 + (individual[0] - 0.5) * 20, 165, 185)
    T2 = np.clip(195 + (individual[1] - 0.5) * 20, 185, 205)
    T3 = np.clip(235 + (individual[2] - 0.5) * 20, 225, 245)
    T4 = np.clip(255 + (individual[3] - 0.5) * 20, 245, 265)
    T5 = 25  # 小温区10~11固定

    if not (T1 < T2 < T3 < T4):
        return 1e6,

    T_set0 = [T1, T2, T3, T4]
    v0 = 65 + 35 * individual[4]

    t_sim = np.arange(0, 700, 0.5)
    x_sim = v0 / 60 * t_sim
    T0_sim = np.array([environment_temperature(T_set0, xi) for xi in x_sim])
    T_sim = simulate_airzone_piecewise(best_params, t_sim, x_sim, T0_sim)

    area = compute_area(t_sim, T_sim)
    if not check(t_sim, T_sim) or area == np.inf:
        return 1e6,
    return area,

def ga_optimize_area_only(best_params, n_gen=60, pop_size=100):
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual_Area, toolbox.attr_float, 5)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_area, best_params=best_params)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=n_gen,
                                   stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    T1 = np.clip(175 + (best_ind[0] - 0.5) * 20, 165, 185)
    T2 = np.clip(195 + (best_ind[1] - 0.5) * 20, 185, 205)
    T3 = np.clip(235 + (best_ind[2] - 0.5) * 20, 225, 245)
    T4 = np.clip(255 + (best_ind[3] - 0.5) * 20, 245, 265)
    T_set0 = [T1, T2, T3, T4]
    v0 = 65 + 35 * best_ind[4]

    t_sim = np.arange(0, 700, 0.5)
    x_sim = v0 / 60 * t_sim
    T0_sim = np.array([environment_temperature(T_set0, xi) for xi in x_sim])
    T_sim = simulate_airzone_piecewise(best_params, t_sim, x_sim, T0_sim)
    area = compute_area(t_sim, T_sim)

    return (T_set0, v0, area, t_sim, T_sim), log

# === 可视化收敛过程 ===
def plot_area_log(log):
    plt.figure()
    plt.plot([entry['min'] for entry in log], 'g')
    plt.xlabel('迭代次数')
    plt.ylabel('面积')
    plt.title('GA 单目标面积优化收敛曲线')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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

n_segments = 11
initial_guess = [1e-6, 30] * n_segments
bounds = [(1e-8, 1e-4), (1, 200)] * n_segments
result = minimize(objective, initial_guess, bounds=bounds)
best_params = result.x
print("使用 GA 进行面积单目标优化...")
(best_T_set, best_v, area_min_ga, t_area, T_area), log_area = ga_optimize_area_only(best_params)

print("最小面积解：")
print("最优速度 =", round(best_v), "cm/min")
print("最优温度设置 =", np.round(best_T_set).astype(int))
print("最小面积 =", round(area_min_ga, 2))
plot_temperature_profile(best_T_set,best_v,best_params)
# 可视化 GA 收敛曲线
plot_area_log(log_area)
