import numpy as np
import pandas as pd
import math


DNA_len = 0             # 染色体长度（即候选槽宽类型数量）
animal_num = 200        # 种群规模
cross_rate = 0.8        # 交叉概率
variation_rate = 0.01   # 变异概率
generator_n = 1000       # 进化代数
penalty = 5000.0         # 未覆盖药盒的罚系数（越大越偏向可行解）



def load_box_data(file_path):

    df = pd.read_excel(file_path, engine='xlrd')
    id_col = '药品编号'
    w_col = '宽(mm)'
    h_col = '高(mm)'

    df = df[[id_col, w_col, h_col]].dropna()
    ids = df[id_col].tolist()
    ws = df[w_col].tolist()
    hs = df[h_col].tolist()
    return ids, ws, hs


def build_candidate_slots(ws, hs):
    candidate_set = set()
    n = len(ws)

    for i in range(n):
        w = float(ws[i])
        h = float(hs[i])

        s_min = w + 2.0

        s_max1 = 2.0 * w - 1e-6

        #侧翻/旋转上界
        denom = math.sqrt(w ** 2 + h ** 2)
        if denom == 0:
            s_max2 = s_max1
        else:
            s_max2 = 2.0 * w * h / denom

        s_max = min(s_max1, s_max2)

        # 若存在合法区间，则取整加入候选集合
        if s_max >= s_min + 1e-6:
            low = math.ceil(s_min)
            high = math.floor(s_max)
            for s in range(low, high + 1):
                candidate_set.add(s)

    candidate_list = sorted(candidate_set)
    return candidate_list


def matrix(ws, hs, slots):

    n = len(ws)
    m = len(slots)
    C = np.zeros((n, m), dtype=int)

    for i in range(n):
        w = float(ws[i])
        h = float(hs[i])
        denom = math.sqrt(w ** 2 + h ** 2)
        if denom == 0:
            limit_side = 2.0 * w
        else:
            limit_side = 2.0 * w * h / denom

        for j in range(m):
            s = float(slots[j])

            cond1 = (s >= w + 2.0)      # 留间隙
            cond2 = (s < 2.0 * w)       # 不并排
            cond3 = (s <= limit_side)   # 不侧翻/旋转

            if cond1 and cond2 and cond3:
                C[i, j] = 1

    return C


def get_fitness(animals, C):
    pop_size = animals.shape[0]
    n, m = C.shape

    fitness = np.zeros(pop_size, dtype=float)

    for idx in range(pop_size):
        gene = animals[idx]
        selected_count = gene.sum()

        cover_matrix = (C & gene)  # 按位与：只有 gene[j]==1 且 C[i,j]==1 时为1
        covered_flag = cover_matrix.any(axis=1)
        uncovered = np.size(covered_flag) - covered_flag.sum()
        fitness[idx] = 1.0 / (1.0 + selected_count + penalty * uncovered)

    return fitness


def select(animals, fitness):
    total_fit = fitness.sum()
    if total_fit == 0:
        idx = np.random.choice(np.arange(animal_num), size=animal_num, replace=True)
    else:
        p = fitness / total_fit
        idx = np.random.choice(np.arange(animal_num), size=animal_num, replace=True, p=p)
    return animals[idx]


def variation(child, variation_rate):
    if np.random.rand() < variation_rate:
        mutate_point = np.random.randint(0, DNA_len)   # 随机选取要变异的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 0->1, 1->0
    return child


def crossover_and_variation(animals, cross_rate):
    new_animals = []
    for father in animals:
        child = father.copy()  # 先复制父本

        # 按 cross_rate 的概率发生交叉
        if np.random.rand() < cross_rate:
            mother = animals[np.random.randint(animal_num)]
            # 在 [0, DNA_len) 中随机选择交叉点
            cross_point = np.random.randint(0, DNA_len)
            child[cross_point:] = mother[cross_point:]

        # 变异
        child = variation(child, variation_rate)

        new_animals.append(child)

    return np.array(new_animals)


def get_result(animals, C, slots, box_ids):
    fitness = get_fitness(animals, C)
    max_index = np.argmax(fitness)
    best_gene = animals[max_index]
    best_fit = fitness[max_index]

    # 统计选用的槽宽类型
    chosen_index = np.where(best_gene == 1)[0]
    chosen_slots = [slots[j] for j in chosen_index]

    # 统计未覆盖药盒
    n, m = C.shape
    cover_matrix = (C & best_gene)
    covered_flag = cover_matrix.any(axis=1)
    uncovered_indices = np.where(~covered_flag)[0]
    uncovered_boxes = [box_ids[i] for i in uncovered_indices]

    print("适应度值:", best_fit)
    print("选用的槽宽类型数量:", len(chosen_slots))
    print("选用的槽宽类型(mm):", chosen_slots)

    if len(uncovered_boxes) == 0:
        print("所有药盒均已被至少一种槽宽类型覆盖。")
    else:
        print("未被覆盖的药盒编号:", uncovered_boxes)

    return best_gene, chosen_slots, uncovered_boxes


if __name__ == "__main__":
    box_ids, ws, hs = load_box_data("附件1-药盒型号.xls")

    slots = build_candidate_slots(ws, hs)
    print("候选槽宽类型数量 m =", len(slots))
    print("候选槽宽类型列表:", slots)

    C = matrix(ws, hs, slots)

    DNA_len = len(slots)

    animals = np.random.randint(2, size=(animal_num, DNA_len))

    for g in range(generator_n):
        fitness = get_fitness(animals, C)
        selected = select(animals, fitness)
        animals = crossover_and_variation(selected, cross_rate)

    best_gene, chosen_slots, uncovered_boxes = get_result(animals, C, slots, box_ids)
