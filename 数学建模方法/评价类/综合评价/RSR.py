# rank_sum_ratio，秩和比法 + 概率单位 Probit 拟合
import numpy as np
import pandas as pd
from scipy.stats import norm


def rsr_method_with_probit(data, weights=None):
    # 假设 data 是 DataFrame，行是评价对象，列是评价指标

    # 步骤1：编秩，默认采用从小到大（效益型指标），值越小排名越靠前
    R = data.rank(ascending=True, method='average')  # 使用非整数编秩法（推荐）

    # 步骤2：加权求秩和比 RSR
    if weights is None:
        weights = np.ones(data.shape[1]) / data.shape[1]  # 默认等权重
    weights = np.array(weights)
    RSR = (R * weights).sum(axis=1) / len(weights)  # 注意除以指标数，得秩和比

    # 步骤3：确定 RSR 的分布信息
    rsr_df = pd.DataFrame({'RSR': RSR})
    rsr_df['频数'] = 1
    rsr_df = rsr_df.groupby('RSR').agg({'频数': 'count'}).reset_index()

    # 计算累计频数、评价对象数、累计频率、秩次平均
    rsr_df['累计频数'] = rsr_df['频数'].cumsum()
    rsr_df['评价对象数'] = len(data)
    rsr_df['评价对象数*100%'] = rsr_df['累计频数'] / rsr_df['评价对象数']

    # Step 4: Probit值（查正态分布函数的分位点）
    n = len(data)
    rsr_df['Probit'] = rsr_df['评价对象数*100%'].apply(
        lambda x: round(norm.ppf(1 - 1 / (4 * n)), 6) if x >= 1.0 else round(norm.ppf(x), 6)
    )

    # Step 5：将 Probit 值映射回到原数据中（用于后续拟合或可视化）
    rsr_full = pd.DataFrame({'RSR': RSR})
    rsr_full = rsr_full.merge(rsr_df[['RSR', 'Probit']], on='RSR', how='left')

    return rsr_full, rsr_df


# 示例数据（可替换成你实际数据）
data = pd.DataFrame({
    '指标1': [10, 15, 12, 8, 13, 11, 16],
    '指标2': [4, 6, 5, 3, 7, 4, 8],
    '指标3': [20, 18, 22, 16, 19, 21, 15]
})

# 示例权重（手动指定或使用熵权法/CRITIC等自动获取）
weights = [0.3, 0.3, 0.4]

# 执行 RSR 分析
rsr_full, rsr_stat = rsr_method_with_probit(data, weights)

# 打印详细结果
print("各评价对象 RSR 及 Probit 值：")
print(rsr_full)

print("\nRSR 分布频率与累计分布表：")
print(rsr_stat)
