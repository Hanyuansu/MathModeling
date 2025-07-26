import pandas as pd
import numpy as np
import math

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
A=np.array([[0,0,-300.79,1]]).T #大地坐标系下的顶点坐标
A=np.linalg.inv(R0)@A
X=round(A[0][0],4)
Y=round(A[1][0],4)
Z=round(A[2][0],4)
df = pd.DataFrame([[X, Y, Z]], columns=['X坐标（米）', 'Y坐标（米）', 'Z坐标（米）'])
with pd.ExcelWriter('附件4.xlsx', mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
    df.to_excel(writer, sheet_name='理想抛物面顶点坐标', index=False, header=False,startrow=1)
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
# 保留满足条件的坐标
adjust_coords=[coord for coord in coords if is_valid(coord)]
# 提取原始 DataFrame 的行号
row_indices = [int(coord[3]) for coord in adjust_coords]
# 根据行号查找对应主索节点编号
adjust_ids = df.iloc[row_indices]["主索节点编号"].values
adjust_coords = np.array(adjust_coords)
# print("符合条件的节点个数:", adjust_coords.shape[0])
# print("对应主索节点编号:", adjust_ids)
m=adjust_coords.shape[0]

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

##验证一下
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

