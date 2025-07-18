import numpy as np  
  
def normalize(matrix, is_benefit):  
    """规范化矩阵  
      
    参数:  
    matrix (numpy.ndarray): 输入的原始数据矩阵  
    is_benefit (list[bool]): 表示各列是否为效益型指标的布尔列表  
      
    返回:  
    numpy.ndarray: 规范化后的矩阵  
    """  
    normalized_matrix = np.zeros_like(matrix)  
    for j in range(matrix.shape[1]):  
        if is_benefit[j]:  
            normalized_matrix[:, j] = (matrix[:, j] - np.min(matrix[:, j])) / (np.max(matrix[:, j]) - np.min(matrix[:, j]))  
        else:  
            normalized_matrix[:, j] = (np.max(matrix[:, j]) - matrix[:, j]) / (np.max(matrix[:, j]) - np.min(matrix[:, j]))  
    return normalized_matrix  
  
def vikor(matrix, weights, is_benefit, v=0.5, p=1):  
    """VIKOR方法  
      
    参数:  
    matrix (numpy.ndarray): 输入的原始数据矩阵  
    weights (numpy.ndarray): 各列的权重  
    is_benefit (list[bool]): 表示各列是否为效益型指标的布尔列表  
    v (float, 可选): 群体效用和个体遗憾之间的权重，默认为0.5  
    p (int, 可选): 距离测度的参数，默认为1  
      
    返回:  
    tuple: 包含综合得分、到正理想解的距离和到负理想解的距离的元组  
    """  
    # 规范化矩阵  
    Z = normalize(matrix, is_benefit)  
    # 加权规范化矩阵  
    Z_weighted = Z * weights  
      
    # 确定正理想解和负理想解  
    Z_star = np.max(Z_weighted, axis=0)  
    Z_nadir = np.min(Z_weighted, axis=0)  
      
    # 计算距离  
    S = np.zeros(matrix.shape[0])  
    R = np.zeros(matrix.shape[0])  
    for i in range(matrix.shape[0]):  
        S[i] = np.sum(np.power(np.abs(Z_weighted[i, :] - Z_star), p)) ** (1/p)  
        R[i] = np.max(np.abs(Z_weighted[i, :] - Z_nadir))  
      
    # 计算得分  
    S_max, S_min = np.max(S), np.min(S)  
    R_max, R_min = np.max(R), np.min(R)  
    Q = v * (S - S_min) / (S_max - S_min) + (1 - v) * (R - R_min) / (R_max - R_min)  
      
    return Q, S, R  
  
# 示例数据  
matrix = np.array([[10, 20, 30], [20, 10, 40], [30, 30, 20]])  
weights = np.array([0.3, 0.3, 0.4])  
is_benefit = [True, True, True]  # 假设所有指标都是效益型  
  
# 调用VIKOR方法  
Q, S, R = vikor(matrix, weights, is_benefit)  
print("综合得分 Q:", Q)  
print("到正理想解的距离 S:", S)  
print("到负理想解的距离 R:", R)  
  
# 可以根据Q进行排序，选择最优方案  
best_index = np.argmin(Q)  
print("最优方案索引:", best_index)