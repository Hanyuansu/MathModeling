import torch
import torch.nn as nn
import  torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import  DataLoader,TensorDataset
import pandas as pd
import numpy as np

#1.加载数据
data_path='./data/附件2(Attachment 2)/Data.xlsx'
data=pd.read_excel(data_path,header=None)
X=data.values.astype(np.float32)
print(f"原始数据形状: {X.shape}")

#2.数据归一化到0~1(匹配sigmoid输出范围)
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X).astype(np.float32)

#3.构建Autoencoder模型
input_dim=X_scaled.shape[1]
encoding_dim=100

class Autoencoder(nn.Module):
    def __init__(self,input_dim,encoding_dim):
        super(Autoencoder,self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(input_dim,512),
            nn.ReLU(),
            nn.Linear(512,encoding_dim),
            nn.ReLU()
        )

        self.decoder=nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self,x):
     encoded=self.encoder(x)
     decoded=self.decoder(encoded)
     return decoded

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Autoencoder(input_dim,encoding_dim).to(device)
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=1e-3)

#4.准备数据加载器
batch_size=256#可调整
dataset=TensorDataset(torch.from_numpy(X_scaled))
dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

#训练模型
num_epochs=100
for epoch in range(num_epochs):
    model.train()
    epoch_loss=0
    for batch in dataloader:
        inputs=batch[0].to(device)
        outputs=model(inputs)
        loss=criterion(outputs,inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()*inputs.size(0)
    epoch_loss/=len(dataset)
    if(epoch+1)%10==0:
        print(f"Epoch[{epoch+1}/{num_epochs}],Loss:{epoch_loss:.6f}")

#6.计算压缩比和存储节省率
compression_ratio=encoding_dim/input_dim
storage_saving_rate=1-compression_ratio

print(f"\n压缩比：{compression_ratio:.2%}")
print(f"存储节省率：{storage_saving_rate:.2%}")


#不同参数压缩比，结果在压缩效果参数搜索结果中
# import torch.optim as optim
# from sklearn.preprocessing import MinMaxScaler
# from torch.utils.data import DataLoader, TensorDataset
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error
# import os
#
# # ===== 1. 读取数据并归一化 =====
# data_path = './data/附件2(Attachment 2)/Data.xlsx'
# data = pd.read_excel(data_path, header=None)
# X = data.values.astype(np.float32)
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X).astype(np.float32)
# input_dim = X.shape[1]
#
# # 转为 PyTorch Tensor
# X_tensor = torch.from_numpy(X_scaled)
# dataset = TensorDataset(X_tensor)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # ===== 2. 定义 Autoencoder 模型类 =====
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, encoding_dim, hidden_dim):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, encoding_dim),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(encoding_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
#
# # ===== 3. 搜索空间设置 =====
# encoding_dims = [10, 20, 50, 80, 100, 150]
# hidden_dims = [128, 256, 512]
# batch_size = 256
# num_epochs = 50
# learning_rate = 1e-3
#
# # ===== 4. 遍历组合进行训练和评估 =====
# results = []
#
# for enc_dim in encoding_dims:
#     for hid_dim in hidden_dims:
#         print(f"\n 正在训练：encoding_dim={enc_dim}, hidden_dim={hid_dim}")
#         model = Autoencoder(input_dim, enc_dim, hid_dim).to(device)
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#         # === 模型训练 ===
#         for epoch in range(num_epochs):
#             model.train()
#             epoch_loss = 0
#             for batch in dataloader:
#                 inputs = batch[0].to(device).float()
#                 outputs = model(inputs)
#                 loss = criterion(outputs, inputs)
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss += loss.item() * inputs.size(0)
#
#         # === 模型评估 ===
#         model.eval()
#         with torch.no_grad():
#             outputs = model(X_tensor.to(device).float()).cpu().numpy()
#         mse = mean_squared_error(X_scaled, outputs)
#         compression_ratio = enc_dim / input_dim
#         storage_saving = 1 - compression_ratio
#
#         results.append({
#             'encoding_dim': enc_dim,
#             'hidden_dim': hid_dim,
#             'MSE': mse,
#             'compression_ratio': compression_ratio,
#             'storage_saving_rate': storage_saving
#         })
#
# # ===== 5. 输出结果为表格 =====
# import pandas as pd
# results_df = pd.DataFrame(results)
# results_df = results_df.sort_values(by='MSE')  # 按误差升序排序
# results_df.to_excel("压缩效果参数搜索结果.xlsx", index=False)
#
# # ===== 6. 推荐最优配置 =====
# best = results_df.iloc[0]
# print("\n 最优结果：")
# print(f"Encoding dim: {best['encoding_dim']}, Hidden dim: {best['hidden_dim']}")
# print(f"MSE: {best['MSE']:.6f}, 压缩比: {best['compression_ratio']:.4f}, 存储节省率: {best['storage_saving_rate']:.2%}")
