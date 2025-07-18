# CRITIC方法：客观方法，求权重

import numpy as np
import pandas as pd
from scipy.stats import shapiro

# 判断一个指标是否服从正态分布（Shapiro-Wilk检验）
def is_normal_distribution(series, alpha=0.05):
    stat, p = shapiro(series)
    return p > alpha

# 步骤：先分布判断再标准化，最后统一
#负向指标与正向指标的处理是相反的
def normalize_column(series):
    if is_normal_distribution(series):
        normed = (series - series.mean()) / series.std(ddof=1)
    else:
        normed = (series - series.min()) / (series.max() - series.min())

    # 再做一次 min-max 统一处理，防止尺度不一致
    min_val, max_val = normed.min(), normed.max()
    if max_val == min_val:
        return pd.Series(0, index=series.index)
    return (normed - min_val) / (max_val - min_val)


# 对整个DataFrame进行正向化处理
def normalize_dataframe(df):
    norm_df = pd.DataFrame()
    for col in df.columns:
        norm_df[col] = normalize_column(df[col])
    return norm_df.fillna(0)

# 计算CRITIC权重（输入应为正向化后的数据）
def compute_critic_weights(df):
    # 步骤一：计算每列标准差（衡量对比强度）
    std_dev = df.std(ddof=1)

    # 步骤二：计算相关系数矩阵
    corr_matrix = df.corr().values

    # 步骤三：计算冲突性系数（1 - |相关系数|）的和
    conflict = np.sum(1 - np.abs(corr_matrix), axis=1)

    # 步骤四：计算每个指标的信息量
    critic_score = std_dev.values * conflict

    # 步骤五：归一化为权重
    weights = critic_score / np.sum(critic_score)
    return pd.Series(weights, index=df.columns)


def compute_weight(data):
    norm_data = normalize_dataframe(data)
    weights = compute_critic_weights(norm_data)
    return weights

# 示例：使用CRITIC方法对4个指标赋权,一列一种指标
data = pd.DataFrame({
    '经济指标': [100, 120, 130, 110, 140],
    '环境指标': [0.3, 0.25, 0.35, 0.28, 0.40],
    '社会指标': [50, 45, 60, 55, 52],
    '资源消耗': [20, 19, 25, 23, 18],
})

# 计算权重
weights = compute_weight(data)

# 输出结果
print("CRITIC权重: ")
print(weights)
print(data)