import numpy as np  
  
def topsis(data, weights):  
    # Step 1: 标准化处理  
    data_normalized = data / np.sqrt((data**2).sum(axis=0, keepdims=True))  
      
    # Step 2: 加权标准化  
    weighted_data = data_normalized * weights

    # Step 3: 确定正理想解和负理想解  
    S_plus = np.max(weighted_data, axis=0)  
    S_minus = np.min(weighted_data, axis=0)

    # Step 4: 计算距离  
    D_plus = np.sqrt(((weighted_data - S_plus)**2).sum(axis=1))  
    D_minus = np.sqrt(((weighted_data - S_minus)**2).sum(axis=1))  
      
    # Step 5: 计算接近程度  
    C_i = D_minus / (D_plus + D_minus)  
      
    # Step 6: 排序  
    rank_order = np.argsort(C_i)[::-1]  # 按接近程度降序排列  
      
    return C_i, rank_order  
  
# 示例数据  
data = np.array([  
    [12, 3, 45],  
    [24, 6, 40],  
    [18, 2, 30],  
    [10, 5, 42]  
])  
  
# 示例权重  
weights = np.array([0.3, 0.2, 0.5])  
  
# 执行TOPSIS  
C_i, rank_order = topsis(data, weights)  
  
# 打印结果  
print("接近程度C_i:", C_i)  
print("排序结果:", rank_order)