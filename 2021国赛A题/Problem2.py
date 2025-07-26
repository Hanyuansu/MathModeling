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

df1=pd.read_excel('附件3.xlsx')
