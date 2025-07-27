import pandas as pd
import numpy as np

# 读取数据
df_opt = pd.read_excel("优化后的主索节点坐标与误差.xlsx")  # 优化后的主索节点坐标
df_base = pd.read_excel("merged.xlsx")  # 基准态主索节点与上端点坐标

# 构建 node_to_idx 和拉索长度 l
df1 = pd.read_excel("附件3.xlsx", usecols=["主索节点1", "主索节点2", "主索节点3"])
nodes = sorted(set(df1["主索节点1"]) | set(df1["主索节点2"]) | set(df1["主索节点3"]))
node_to_idx = {node: idx for idx, node in enumerate(nodes)}

# 初始化拉索长度数组
l = np.zeros(len(nodes))
for _, row in df_base.iterrows():
    idx = node_to_idx[row['主索节点编号']]
    x, y, z = row['X坐标（米）'], row['Y坐标（米）'], row['Z坐标（米）']
    xc, yc, zc = row['基准态时上端点X坐标（米）'], row['基准态时上端点Y坐标（米）'], row['基准态时上端点Z坐标（米）']
    l[idx] = np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2)

# 反解促动器上端点
results = []
for _, row in df_opt.iterrows():
    node_id = row['主索节点编号']
    if node_id not in node_to_idx:
        continue

    idx = node_to_idx[node_id]
    li = l[idx]

    x, y, z = row['X坐标'], row['Y坐标'], row['Z坐标']

    # 查找对应基准态上端点
    row_base = df_base[df_base['主索节点编号'] == node_id]
    if row_base.empty:
        continue

    x0 = row_base['基准态时上端点X坐标（米）'].values[0]
    y0 = row_base['基准态时上端点Y坐标（米）'].values[0]
    z0 = row_base['基准态时上端点Z坐标（米）'].values[0]

    # 单位方向向量（从球心指向基准态上端点）
    norm = np.sqrt(x0**2 + y0**2 + z0**2)
    ux, uy, uz = x0 / norm, y0 / norm, z0 / norm

    # 构造一元二次方程：a λ^2 + b λ + c = 0
    dx, dy, dz = x - x0, y - y0, z - z0

    a = 1  # 单位向量模长为1
    b = -2 * (dx * ux + dy * uy + dz * uz)
    c = dx**2 + dy**2 + dz**2 - li**2

    Δ = b**2 - 4 * a * c
    if Δ < 0:
        continue  # 无解跳过

    λ1 = (-b + np.sqrt(Δ)) / (2 * a)
    λ2 = (-b - np.sqrt(Δ)) / (2 * a)
    λ = max(λ1, λ2)

    # 解得上端点新位置
    xc = x0 + λ * ux
    yc = y0 + λ * uy
    zc = z0 + λ * uz

    l_base = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    l_new = np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2)
    delta = l_new - l_base

    results.append([
        node_id, x, y, z,
        xc, yc, zc,
        li, l_base, l_new, delta
    ])

# 保存结果
df_result = pd.DataFrame(results, columns=[
    "主索节点编号", "主索X", "主索Y", "主索Z",
    "上端点X", "上端点Y", "上端点Z",
    "目标长度", "原始长度", "当前长度", "伸缩量"
])
df_result.to_excel("反解_促动器上端点坐标_及伸缩量.xlsx", index=False)
