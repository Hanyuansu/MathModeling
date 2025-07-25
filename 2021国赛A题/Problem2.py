import pandas as pd
import numpy as np
import math

#1.构建旋转矩阵
a=np.radians(36.795)
b=np.radians(90-78.169)
Rz=np.array([[math.cos(a),math.sin(a),0],
            [-math.sin(a),math.cos(a),0],
            [0,0,1]])
Ry=np.array([[math.cos(b),0,-math.sin(b)],
            [0,1,0],
            [math.sin(b),0,math.cos(b)]])
R0=Ry@Rz
print(R0)
#2.求天体坐标系下的顶点坐标
A=np.array([[0,0,-300.79]]).T #大地坐标系下的顶点坐标
A=np.linalg.inv(R0)@A
print(A)

#3.计算应调整的索网节点
R=300.4 #半径
f=0.466*R+0.39 #焦距
#读坐标
df=pd.read_excel("merged.xlsx")
x = df['X坐标（米）'].values
y = df['Y坐标（米）'].values
z = df['Z坐标（米）'].values
mat = np.column_stack((x, y, z)).T
print(mat)

#旋转
mat=R0@mat #天体坐标系下的主索节点坐标
mat=mat.T
print(mat)

#筛选应该调整的主索节点
def is_valid(coord):
    x,y,z=coord
    return

