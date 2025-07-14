import pandas as pd
import numpy as np
from keras.src.losses import mean_squared_error
from sklearn.preprocessing import  StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from bayes_opt import BayesianOptimization

#1.加载数据
X_path='./Data/附件5(Attachment 5)/5-X.xlsx'
Y_path='./Data/附件5(Attachment 5)/5-Y.xlsx'

X=pd.read_excel(X_path,header=None).values.astype(np.float32)
Y=pd.read_excel(Y_path,header=None).values.flatten().astype(np.float32)

print(f"X形状:{X.shape},Y形状:{Y.shape}")

#2.标准化
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#转为Torch Tensor
X_tensor=torch.from_numpy(X_scaled)
Y_tensor=torch.from_numpy(Y).view(-1,1)

#3.定义Autoencoder模型
class Autoencoder(nn.Module):
    def __init__(self,input_dim,encoding_dim):
        super(Autoencoder,self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Linear(128,encoding_dim),
            nn.ReLU()
        )

        self.decoder=nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self,x):
     encoded=self.encoder(x)
     decoded=self.decoder(encoded)
     return decoded,encoded

#4.定义MLP模型
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

#5.定义训练加评估函数
def train_evaluate(encoding_dim,mlp_hidden_dim,mlp_hidden_layers,lr,batch_size):
    encoding_dim=int(encoding_dim)
    mlp_hidden_dim=int(mlp_hidden_dim)
    mlp_hidden_layers=int(mlp_hidden_layers)
    batch_size=int(batch_size)
    lr=float(lr)

    autoencoder=Autoencoder(X.shape[1],encoding_dim)
    mlp=MLPRegressor(encoding_dim,mlp_hidden_dim,mlp_hidden_layers)

    ae_criterion=nn.MSELoss()
    mlp_criterion=nn.MSELoss()

    ae_optimizer=optim.Adam(autoencoder.parameters(),lr=lr)
    mlp_optimizer=optim.Adam(mlp.parameters(),lr=lr)

    dataset=torch.utils.data.TensorDataset(X_tensor,Y_tensor)
    dataloader=torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)

    autoencoder.train()
    mlp.train()

    for epoch in range(50):
        for xb,yb in dataloader:
            #训练autoencoder
            ae_optimizer.zero_grad()
            decoded,encoded=autoencoder(xb)
            ae_loss=ae_criterion(decoded,xb)
            ae_loss.backward()
            ae_optimizer.step()

            #训练MLP
            mlp_optimizer.zero_grad()
            encoded_detatch=encoded.detach()
            pred=mlp(encoded_detatch)
            mlp_loss=mlp_criterion(pred,yb)
            mlp_loss.backward()
            mlp_optimizer.step()

    #评估
    autoencoder.eval()
    mlp.eval()
    with torch.no_grad():
        _,encoded_all=autoencoder(X_tensor)
        pred_all=mlp(encoded_all).numpy().flatten()
    mse=mean_squared_error(Y,pred_all)
    print(f"enc_dim={encoding_dim},mlp_hidden={mlp_hidden_dim},mlp_layer={mlp_hidden_layers},lr={lr:.5f},batch_size={batch_size},MSE={mse:.6f}")
    return -mse#贝叶斯最大优化器目标，取负数则为最小化
#5.定义贝叶斯优化器
pbounds={
    'encoding_dim':(10,50),
    'mlp_hidden_dim':(32,256),
    'mlp_hidden_layers':(1,4),
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