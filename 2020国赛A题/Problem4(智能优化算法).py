import math

from deap import base, creator, tools, algorithms
import random
import numpy as np
import matplotlib.pyplot as plt
from Problem1 import boundary, environment_temperature, segment_index, simulate_airzone_piecewise, objective
from Problem2 import check
from scipy.optimize import minimize

# === 对称性指标函数（改进版：使用左右半峰宽差异衡量） ===
def compute_symmetry(t_sim, T_sim):
    idx_217 = np.where(T_sim >= 217)[0]
    if len(idx_217) < 2:
        return np.inf
    t3 = idx_217[0]
    t4 = idx_217[-1]
    tm = np.argmax(T_sim)
    if tm <= t3 or tm >= t4:
        return np.inf
    # 对称性新定义：左右半峰宽差异（越小越对称）
    left_width = tm - t3
    right_width = t4 - tm
    return (left_width - right_width) ** 2

# === 面积指标函数 ===
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

# === 创建遗传算法适应度函数（组合优化） ===
creator.create("FitnessMin_Combo", base.Fitness, weights=(-1.0,))
creator.create("Individual_Combo", list, fitness=creator.FitnessMin_Combo)

def evaluate_combined(individual, best_params, area_min, sym_min):
    T1 = np.clip(175 + (individual[0] - 0.5) * 20, 165, 185)
    T2 = np.clip(195 + (individual[1] - 0.5) * 20, 185, 205)
    T3 = np.clip(235 + (individual[2] - 0.5) * 20, 225, 245)
    T4 = np.clip(255 + (individual[3] - 0.5) * 20, 245, 265)

    if not (T1 < T2 < T3 < T4):
        return 1e12,

    T_set0 = [T1, T2, T3, T4]
    v0 = 65 + 35 * individual[4]

    t_sim = np.arange(0, 700, 0.5)
    x_sim = v0 / 60 * t_sim
    T0_sim = np.array([environment_temperature(T_set0, xi) for xi in x_sim])
    T_sim = simulate_airzone_piecewise(best_params, t_sim, x_sim, T0_sim)

    area = compute_area(t_sim, T_sim)
    sym = compute_symmetry(t_sim, T_sim)

    if not check(t_sim, T_sim) or area == np.inf or sym == np.inf:
        return 1e12,

    # 若任何目标函数超过其单目标最优值，则淘汰
    penalty = 0
    if area > area_min:
        penalty += (area - area_min) / area_min
    if sym > sym_min:
        penalty += (sym - sym_min) / sym_min

    score = area / area_min + sym / sym_min + 10 * penalty  # 可调惩罚系数
    return score,
    # if area > area_min or sym > sym_min:
    #     return 1e12,
    #
    # score = area / area_min + sym / sym_min
    # return score,

def ga_optimize_combined(best_params, area_min, sym_min, n_gen=60, pop_size=100):
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual_Combo, toolbox.attr_float, 5)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_combined, best_params=best_params, area_min=area_min, sym_min=sym_min)
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
    sym = compute_symmetry(t_sim, T_sim)
    score = area / area_min + sym / sym_min

    return (T_set0, v0, area, sym, t_sim, T_sim, score), log

# === 可视化收敛过程 ===
def plot_combined_log(log):
    plt.figure()
    plt.plot([entry['min'] for entry in log], 'k')
    plt.xlabel('迭代次数')
    plt.ylabel('组合目标函数值')
    plt.title('GA 组合目标优化收敛曲线')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === 主调用逻辑（示例） ===
if __name__ == '__main__':
    print("拟合最优传热参数...")
    initial_guess = [1e-6, 30] * 11
    bounds = [(1e-8, 1e-4), (1, 200)] * 11
    result = minimize(objective, initial_guess, bounds=bounds)
    best_params = result.x

    print("搜索最小面积...")
    area_min = optimize_single_objective_area(best_params)
    print("搜索最小对称性...")
    sym_min = optimize_single_objective_sym(best_params)
    print("执行组合优化...")
    (best_T, best_v, best_area, best_sym, t_sim, T_sim, score), log = ga_optimize_combined(best_params, area_min, sym_min)

    print("\n=== 最终组合优化结果 ===")
    print("速度:", round(best_v), "cm/min")
    print("温度设置:", np.round(best_T).astype(int))
    print("最小面积:", round(best_area, 2))
    print("最小对称性:", math.sqrt(round(best_sym, 2)))
    print("目标函数值:", round(score, 4))

    plot_combined_log(log)
