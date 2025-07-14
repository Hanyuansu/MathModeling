import pandas as pd
import numpy as np
from keras.src.losses import mean_squared_error
from sklearn.preprocessing import  StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from bayes_opt import BayesianOptimization

#1.加载数据
X_path='./Data/附件4(Attachment 4)/4-X.xlsx'
Y_path='./Data/附件4(Attachment 4)/4-Y.xlsx'

X=pd.read_excel(X_path,header=None).values.astype(np.float32)
Y=pd.read_excel(Y_path,header=None).values.flatten().astype(np.float32)

print(f"X形状:{X.shape},Y形状:{Y.shape}")

#2.标准化
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#转为Torch Tensor
X_tensor=torch.from_numpy(X_scaled)
Y_tensor=torch.from_numpy(Y).view(-1,1)

#3.定义MLP模型
class MLPRegressor(nn.Module):
    def __init__(self,input_dim,hidden_dim,hidden_layers):
        super(MLPRegressor,self).__init__()
        layers=[nn.Linear(input_dim,hidden_dim),nn.ReLU()]
        for _ in range(hidden_layers-1):
            layers.append(nn.Linear(hidden_dim,hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim,1))
        self.model=nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)

#4.定义训练加评估函数
def train_evaluate(hidden_dim,hidden_layers,lr,batch_size):
    hidden_dim=int(hidden_dim)
    hidden_layers=int(hidden_layers)
    batch_size=int(batch_size)
    lr=float(lr)

    model=MLPRegressor(X.shape[1],hidden_dim,hidden_layers)
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)

    dataset=torch.utils.data.TensorDataset(X_tensor,Y_tensor)
    dataloader=torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)

    model.train()
    for epoch in range(50):
        for xb,yb in dataloader:
            optimizer.zero_grad()
            pred=model(xb)
            loss=criterion(pred,yb)
            loss.backward()
            optimizer.step()

    #评估MSE
    model.eval()
    with torch.no_grad():
        pred=model(X_tensor).numpy().flatten()
    mse=mean_squared_error(Y,pred)
    print(f"hidden_dim={hidden_dim},hidden_layers={hidden_layers},lr={lr:.5f},batch_size={batch_size},MSE={mse:.6f}")
    return -mse#贝叶斯最大优化器目标，取负数则为最小化
#5.定义贝叶斯优化器
pbounds={
    'hidden_dim':(32,256),
    'hidden_layers':(1,4),
    'lr':(1e-4,1e-2),
    'batch_size':(32,256)
}

optimizer=BayesianOptimization(
    f=train_evaluate,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

#6.执行优化
optimizer.maximize(init_points=5,n_iter=15)

#最佳结果
print(f"\n最佳参数:{optimizer.max['params']}")
print(f"\n最佳(负)MSE得分:{optimizer.max['target']:.6f}")
print(f"\n最佳MSE得分:{-optimizer.max['target']:.6f}")




