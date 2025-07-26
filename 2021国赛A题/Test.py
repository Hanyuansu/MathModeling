import pandas as pd
import numpy as np

# 只读取主索节点列
df = pd.read_excel("附件3.xlsx", usecols=["主索节点1", "主索节点2", "主索节点3"])

# 所有唯一主索节点编号
nodes = sorted(set(df["主索节点1"]) | set(df["主索节点2"]) | set(df["主索节点3"]))
node_to_idx = {node: idx for idx, node in enumerate(nodes)}
n = len(nodes)

# 初始化邻接矩阵
adj_matrix = np.zeros((n, n), dtype=int)

# 构建无向图边关系（三元组两两连边）
for _, row in df.iterrows():
    a, b, c = row["主索节点1"], row["主索节点2"], row["主索节点3"]
    idxs = [node_to_idx[a], node_to_idx[b], node_to_idx[c]]
    for i in range(3):
        for j in range(i + 1, 3):
            adj_matrix[idxs[i]][idxs[j]] = 1
            adj_matrix[idxs[j]][idxs[i]] = 1
# 构造邻接矩阵 DataFrame
adj_df = pd.DataFrame(adj_matrix, index=nodes, columns=nodes)

# 输出前几行看结果
print(adj_df.head())

# 如果你想保存为 Excel 文件：
# adj_df.to_excel("邻接矩阵.xlsx")
