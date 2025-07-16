#层次分析法，主观方法
#参考：https://blog.csdn.net/qq_41686130/article/details/122081827
import numpy as np

#计算一致性指标CI
def consistency_index(n,lambda_max):
    return (lambda_max-n) / (n-1)

#一致性指标RI
def random_index(n):
    #RI表
    RI_values = {  
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,  
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49  
    }  
    return RI_values[n]  

#计算一致性比例CR
def consistency_ratio(CI,RI):
    return CI / RI

#计算判断矩阵的特征向量和权重，三种方法
def compute_weights(A,method):
    #1.特征向量法
    if(method =='eigen'):
        # 计算判断矩阵的特征值和特征向量  
        eigenvalues, eigenvectors = np.linalg.eig(A)  
        # 最大的特征值对应的特征向量即为权重向量  
        max_eigenvalue_index = np.argmax(eigenvalues)  
        weights = eigenvectors[:, max_eigenvalue_index].real  
        # 归一化权重向量  
        weights = weights / np.sum(weights)  
        return weights  
    
    #2.几何平均
    elif(method =='geometric'):
        n = A.shape[0]
        # 每一行取几何平均值
        row_product = np.prod(A, axis=1)  # 按行求积
        geo_mean = row_product ** (1 / n)  # n 次方根
        # 归一化处理，得到权重向量
        weights = geo_mean / np.sum(geo_mean)
        return weights
    
    #3.算数平均
    else:
        # 按列归一化
        col_sum = np.sum(A, axis=0) # 每列求和
        norm_matrix = A / col_sum # 每个元素除以所在列的总和
        # 按行求平均（算术平均）
        weights = np.mean(norm_matrix, axis=1)
        # 归一化，确保所有权重之和为 1
        weights = weights / np.sum(weights)
        return weights

# 计算一致性检验  
def check_consistency(A, weights):  
    n = A.shape[0]  
    # 计算判断矩阵的最大特征值  
    lambda_max = np.dot(np.dot(A, weights.T), weights) / np.sum(weights**2)  
    # 计算一致性指标CI  
    CI = consistency_index(n, lambda_max)  
    # 查找随机一致性指标RI  
    RI = random_index(n)  
    # 计算一致性比例CR  
    CR = consistency_ratio(CI, RI)  
    return CR  
  
# 示例：一个3x3的判断矩阵
# 实际应用替换成对应的判断矩阵即可
A = np.array([[1, 2, 3], [1/2, 1, 2], [1/3, 1/2, 1]])  
  
# 计算权重  
weights = compute_weights(A,'geometric')  
print("Weights:", weights)  
  
# 一致性检验  
CR = check_consistency(A, weights)  
print("Consistency Ratio (CR):", CR)  
if CR < 0.1:  
    print("矩阵通过了一致性检验")  
else:  
    print("矩阵没有通过一致性检验")


