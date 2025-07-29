import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

#1.构建旋转矩阵,最后一列用于记录编号
a=np.radians(36.795)
b=np.radians(90-78.169)
Rz = np.array([
    [ math.cos(a), math.sin(a), 0, 0],
    [-math.sin(a), math.cos(a), 0, 0],
    [ 0          , 0          , 1, 0],
    [ 0,           0          , 0, 1]
])
Ry = np.array([
    [ math.cos(b), 0, -math.sin(b), 0],
    [ 0,           1,  0,           0],
    [ math.sin(b), 0,  math.cos(b), 0],
    [ 0,           0,  0,           1]
])
R0=Ry@Rz
#print(R0)

#2.求天体坐标系下的顶点坐标
A=np.array([[0,0,-300.79,1]]).T
A=np.linalg.inv(R0)@A #大地坐标系下的顶点坐标
X=round(A[0][0],4)
Y=round(A[1][0],4)
Z=round(A[2][0],4)
#print(A)

#3.计算应调整的索网节点
R=300.4 #半径
f=0.466*R+0.39 #焦距
r=150.0
p=4*f
#读坐标
df=pd.read_excel("merged.xlsx")
x = df['X坐标（米）'].values
y = df['Y坐标（米）'].values
z = df['Z坐标（米）'].values
index=np.arange(0,2226)
coords = np.column_stack((x, y, z,index)).T
#print(coords)

#旋转
coords=R0@coords #天体坐标系下的主索节点坐标
coords=coords.T
print(coords)

#筛选应该调整的主索节点
def is_valid(coord):
    x,y,z,index=coord
    return (np.square(x)+np.square(y))*(np.square(r*r/p-R-0.39)+np.square(r)) <= np.square(R*r)
adjust_coords = []
adjust_indices = []
for i, coord in enumerate(coords):
    if is_valid(coord):
        adjust_coords.append(coord)
        adjust_indices.append(i)

#4.构造邻接矩阵
df1 = pd.read_excel("附件3.xlsx", usecols=["主索节点1", "主索节点2", "主索节点3"])
# 去重排序
nodes = sorted(set(df1["主索节点1"]) | set(df1["主索节点2"]) | set(df1["主索节点3"]))
node_to_idx = {node: idx for idx, node in enumerate(nodes)}
n = len(nodes)
adj_matrix = np.zeros((n, n), dtype=int)
for _, row in df1.iterrows():
    a, b, c = row["主索节点1"], row["主索节点2"], row["主索节点3"]
    idxs = [node_to_idx[a], node_to_idx[b], node_to_idx[c]]
    for i in range(3):
        for j in range(i + 1, 3):
            adj_matrix[idxs[i]][idxs[j]] = 1
            adj_matrix[idxs[j]][idxs[i]] = 1
adj_df = pd.DataFrame(adj_matrix, index=nodes, columns=nodes)
#print(adj_df.head())

#5.计算下拉索长度
# adjust_coords是需要调整的主索节点
# node_to_idx是主索节点到邻接矩阵编号的映射
l=np.zeros(n,dtype=float)
for _, row in df.iterrows():
    i=node_to_idx[row['主索节点编号']] #对应到排序中的位置
    x=row['X坐标（米）']
    y=row['Y坐标（米）']
    z=row['Z坐标（米）']
    xc=row['基准态时上端点X坐标（米）']
    yc=row['基准态时上端点Y坐标（米）']
    zc=row['基准态时上端点Z坐标（米）']
    l[i]=np.sqrt((x-xc)**2+(y-yc)**2+(z-zc)**2)
# print(l)

# #验证一下
# L=[]
# for adjust_coord in adjust_coords:
#     x,y,z,index=adjust_coord
#     row_indices = int(index)
#     row=df.iloc[row_indices]
#     i=node_to_idx[row['主索节点编号']]
#     xc=row['基准态时上端点X坐标（米）']
#     yc=row['基准态时上端点Y坐标（米）']
#     zc=row['基准态时上端点Z坐标（米）']
#     coord=np.array([[xc,yc,zc,1]]).T
#     coord=(R0@coord)
#     xc,yc,zc,index=coord
#     if(abs(np.sqrt((x-xc)**2+(y-yc)**2+(z-zc)**2))-l[i]<=0.0001):
#         L.append(1)
# print(len(L))

#6.根据主索节点坐标计算径向到理想抛物面的坐标
def project_to_paraboloid(xp, yp, zp, p):
    r2 = xp**2 + yp**2
    #xp^2+yp^2==0
    if r2 < 1e-10:
        return 0.0, 0.0, -300.79
    a = zp / np.sqrt(r2)
    #xp^2==0
    if abs(xp) < 1e-10:
        z_p = p * a ** 2 + a * np.sqrt(p ** 2 * a ** 2 + 2 * p * (R + 0.39))
        x_p = 0.0
        y_p = np.copysign(np.sqrt(2 * p * (z_p + R +0.39)), yp)
    else:
        b = yp / xp
        z_p = p * a ** 2 + a * np.sqrt(p ** 2 * a ** 2 + 2 * p * (R + 0.39))
        x_sq = 2 * p * (z_p + (R + 0.39)) / (1 + b ** 2)
        x_p = np.copysign(np.sqrt(max(x_sq, 0.0)), xp)
        y_p = b * x_p
    return x_p, y_p, z_p

x0 = coords[adjust_indices, :3].reshape(-1)

# 目标函数
def objective(X):
    X_mat = coords[:, :3].copy()
    X_mat[adjust_indices] = X.reshape(-1, 3)
    total = 0
    for idx in adjust_indices:
        x, y, z = X_mat[idx]
        xp, yp, zp = project_to_paraboloid(x, y, z, p / 2)
        total += 1e6*((x - xp)**2 + (y - yp)**2 + (z - zp)**2)
    return total

# 下拉索长度约束
def stretch_constraint(X):
    X_mat = X.reshape(-1, 3)
    constraints = []
    for idx, adj_coord in enumerate(adjust_coords):
        x, y, z = X_mat[idx]
        row_idx = int(adj_coord[3])
        row = df.iloc[row_idx]
        up_coord = np.array([[
            row['基准态时上端点X坐标（米）'],
            row['基准态时上端点Y坐标（米）'],
            row['基准态时上端点Z坐标（米）'], 1
        ]]).T
        up_coord = R0 @ up_coord
        xc, yc, zc = up_coord[:3, 0]
        D = np.sqrt(xc**2 + yc**2 + zc**2)

        l_current = np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2)

        xs, ys, zs = (D - 0.6) / D * np.array([xc, yc, zc])
        xx, yx, zx = (D + 0.6) / D * np.array([xc, yc, zc])

        l_min = np.sqrt((xs - x)**2 + (ys - y)**2 + (zs - z)**2)
        l_max = np.sqrt((xx - x)**2 + (yx - y)**2 + (zx - z)**2)

        constraints.append(l_current - l_min)
        constraints.append(l_max - l_current)

    return np.array(constraints)

# 反射面板边长约束
edge_list = []
for i in range(n):
    for j in range(i+1, n):
        if adj_matrix[i, j] == 1:
            edge_list.append((i, j))
edge_list = np.array(edge_list)

# 原始边长
e0 = [np.linalg.norm(coords[i, :3] - coords[j, :3]) for i, j in edge_list]
def edge_constraint(X):
    X_mat = coords[:, :3].copy()
    X_mat[adjust_indices] = X.reshape(-1, 3)
    constraints = []
    for idx, (i, j) in enumerate(edge_list):
        e_new = np.linalg.norm(X_mat[i] - X_mat[j])
        rel_err = abs(e_new - e0[idx]) / e0[idx]
        constraints.append(0.0007 - rel_err)
    return np.array(constraints)

# 数值梯度函数（更快）
def numerical_gradient(func, X, eps=1e-3):
    grad = np.zeros_like(X)
    for i in range(len(X)):
        X1, X2 = X.copy(), X.copy()
        X1[i] += eps
        X2[i] -= eps
        grad[i] = (func(X1) - func(X2)) / (2 * eps)
    return grad

#优化
res = minimize(
    objective, x0, method='trust-constr',
    jac=lambda X: numerical_gradient(objective, X),
    constraints=[
        {'type': 'ineq', 'fun': stretch_constraint},
        {'type': 'ineq', 'fun': edge_constraint}
    ],
    options={'verbose': 3, 'maxiter': 1}
)

X_opt = coords[:, :3].copy()
X_opt[adjust_indices] = res.x.reshape(-1, 3)

# 计算误差
errors = []
for idx in adjust_indices:
    x, y, z = X_opt[idx]
    xp, yp, zp = project_to_paraboloid(x, y, z, p/2)
    d = np.sqrt((x - xp)**2 + (y - yp)**2 + (z - zp)**2)
    errors.append(d)

errors = np.array(errors)
print("最大误差:", errors.max())
print("最小误差:", errors.min())
print("平均误差:", errors.mean())
print("误差标准差:", errors.std())

# 导出优化后结果
adjust_node_ids = [df.iloc[int(coords[idx][3])]["主索节点编号"] for idx in adjust_indices]
df_result = pd.DataFrame(X_opt[adjust_indices], columns=["X坐标（米）", "Y坐标（米）", "Z坐标（米）"])
df_result["主索节点编号"] = adjust_node_ids
df_result["误差"] = errors
df_result.to_excel("优化后的主索节点坐标与误差.xlsx", index=False)

# 误差可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, edgecolor='black', color='lightblue')
plt.xlabel('误差 (m)')
plt.ylabel('节点数')
plt.title('优化后主索节点误差分布')
plt.grid(True)
plt.show()

df_opt = pd.read_excel("优化后的主索节点坐标与误差.xlsx")
result1=[]
for _, row in df_opt.iterrows():
    node_id = row["主索节点编号"]
    x_opt, y_opt, z_opt = row["X坐标（米）"], row["Y坐标（米）"], row["Z坐标（米）"]
    opt_coord=np.linalg.inv(R0)@np.array([[x_opt,y_opt,z_opt,1]]).T
    x_opt,y_opt,z_opt=opt_coord[:3,0]
    result1.append({
        "节点编号": node_id,
        "X坐标（米）": round(x_opt, 4),
        "Y坐标（米）": round(y_opt, 4),
        "Z坐标（米）": round(z_opt, 4),
    })


result2 = []
for _, row in df_opt.iterrows():
    node_id = row["主索节点编号"]
    x_d, y_d, z_d = row["X坐标（米）"], row["Y坐标（米）"], row["Z坐标（米）"]

    # 找到基准态上端点坐标
    match = df[df["主索节点编号"] == node_id].iloc[0]
    x_c, y_c, z_c = match["基准态时上端点X坐标（米）"], match["基准态时上端点Y坐标（米）"], match["基准态时上端点Z坐标（米）"]
    l_i = l[node_to_idx[node_id]]  # 下拉索原始长度

    down_coord = np.array([[x_d, y_d, z_d, 1]]).T
    px, py, pz = down_coord[:3, 0]

    up_coord = np.array([[x_c, y_c, z_c, 1]]).T
    up_coord = R0 @ up_coord
    cx, cy, cz = up_coord[:3, 0]

    A = cx**2 + cy**2 + cz**2
    B = cx*px + cy*py + cz*pz
    C = px**2 + py**2 + pz**2

    # 解二次方程 Aλ² - 2Bλ + (C - l²) = 0
    a = A
    b = -2 * B
    c = C - l_i**2
    delta = b**2 - 4*a*c

    if delta < 0:
        print(f"节点 {node_id} 无实数解，跳过")
        continue

    sqrt_delta = np.sqrt(delta)
    lambda1 = (-b + sqrt_delta) / (2*a)
    lambda2 = (-b - sqrt_delta) / (2*a)

    # 选择离1最近的 λ
    lam = lambda1 if abs(lambda1 - 1) < abs(lambda2 - 1) else lambda2
    D_i = np.sqrt(A)
    delta_i = D_i * (1 - lam)

    # 反解上端点坐标
    xd_up, yd_up, zd_up = lam * cx, lam * cy, lam * cz

    result2.append({
        "对应主索节点编号": node_id,
        "伸缩量（米）": round(delta_i,4)
    })

df_ideal_vertex = pd.DataFrame([[X, Y, Z]], columns=['X坐标（米）', 'Y坐标（米）', 'Z坐标（米）'])
df_adjusted_nodes = pd.DataFrame(result1, columns=['节点编号', 'X坐标（米）', 'Y坐标（米）', 'Z坐标（米）'])
df_actuator_delta = pd.DataFrame(result2, columns=['对应主索节点编号', '伸缩量（米）'])
output_path = '附件4.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_ideal_vertex.to_excel(writer, sheet_name='理想抛物面顶点坐标', index=False)
    df_adjusted_nodes.to_excel(writer, sheet_name='调整后主索节点编号及坐标', index=False)
    df_actuator_delta.to_excel(writer, sheet_name='促动器顶端伸缩量', index=False)


