#Entropy Weight Method，熵权法，客观方法，求权重
#可以将熵权法与其他赋权方法结合，如 AHP（层次分析法）或主成分分析法，以更好地解决复杂的决策问题。
import numpy as np

def entropy_weight(data):  
    # 标准化处理 ，可以采用不同的方法
    data_normalized = data / data.sum(axis=0, keepdims=True)  
      
    # 计算熵值  
    m, n = data_normalized.shape  
    epsilon = 1e-10  # 防止log(0)的情况出现  
    entropy = -1 / np.log(m) * np.sum(data_normalized * np.log(data_normalized + epsilon), axis=0)  
     # 1 / np.log(m)，归一化系数，熵的理论最大值是sum(ln(m)*1/m)=ln(m)，所有信息均匀分布
    # 计算差异度  
    d = 1 - entropy  
      
    # 计算权重  
    weight = d / d.sum()  
      
    return weight  
  
# 示例数据
data = np.array([
    [12, 3, 45],
    [24, 6, 40],
    [18, 2, 30],
    [10, 5, 42]
])

  
# 熵权法计算权重  
weights = entropy_weight(data)  
  
# 打印结果  
print("熵权法计算得到的权重:", weights)