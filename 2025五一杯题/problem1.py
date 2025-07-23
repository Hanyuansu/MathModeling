import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader,TensorDataset,random_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import os

#1.读取数据
folder='./Data/附件1(Attachment 1)'
A_path=os.path.join(folder,'A.xlsx')
B_path=os.path.join(folder,'B.xlsx')

A_df=pd.read_excel(A_path,header=None)#第一行默认是列名,没有被计入
B_df=pd.read_excel(B_path,header=None)

A=A_df.values.astype(np.float32)
B=B_df.values.astype(np.float32)

#转成tensor格式，中间要加一个维度,用于后面的CNN
X=torch.tensor(A).unsqueeze(1) #shape(10000,1,99)
Y=torch.tensor(B) #shape(10000,1)

#2.构造数据集和数据加载器
dataset=TensorDataset(X,Y)
train_size=int(0.8*len(dataset))
val_size=len(dataset)-train_size
train_dataset,val_dataset=random_split(dataset,[train_size,val_size])

train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=64)

#定义模型
class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        self.conv1=nn.Conv1d(1,16,kernel_size=3,padding=1)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv1d(16,32,kernel_size=3,padding=1)
        self.dropout=nn.Dropout(0.5)
        self.fc1=nn.Linear(32*99,64)
        self.fc2=nn.Linear(64,1)

    def forward(self,x):
        x=self.relu(self.conv1(x))
        x=self.relu(self.conv2(x))
        x=x.view(x.size(0),-1)#flatten
        x=self.dropout(x)
        x=self.relu(self.fc1(x))
        x=self.fc2(x)
        return x

#4.训练设置
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=CNNRegressor().to(device)

criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)#Adam优化

patience=10
best_val_loss=np.inf
patience_counter=0

train_loss_history=[]
val_loss_history=[]

#5.训练循环
for epoch in range(100):
    model.train()
    train_losses=[]
    for batch_X,batch_Y in train_loader:
        batch_X,batch_Y=batch_X.to(device),batch_Y.to(device)
        optimizer.zero_grad()
        outputs=model(batch_X)
        loss=criterion(outputs,batch_Y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses=[]
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs=model(batch_X)
            loss=criterion(outputs,batch_Y)
            val_losses.append(loss.item())

    avg_train_loss=np.mean(train_losses)
    avg_val_loss=np.mean(val_losses)
    train_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    if avg_val_loss<best_val_loss:
        best_val_loss=avg_val_loss
        patience_counter=0
        best_model_state=model.state_dict()
    else:
        patience_counter+=1
        if patience_counter>=patience:
            print("Early stepping triggering.")
            break

#6.评估指标
model.load_state_dict(best_model_state)
model.eval()

all_preds=[]
all_targets=[]
with torch.no_grad():
    for batch_X, batch_Y in val_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        outputs = model(batch_X).cpu().numpy()
        all_preds.append(outputs)
        all_targets.append(batch_Y.numpy())

preds=np.vstack(all_preds)
targets=np.vstack(all_targets)

mse=mean_squared_error(targets,preds)
mae=mean_absolute_error(targets,preds)
r2=r2_score(targets,preds)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}")



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline


# 1. 数据加载
def load_data():
    folder = './Data/附件1(Attachment 1)'
    A_path = os.path.join(folder, 'A.xlsx')
    B_path = os.path.join(folder, 'B.xlsx')

    A_df = pd.read_excel(A_path, header=None)
    B_df = pd.read_excel(B_path, header=None)

    A = A_df.values.astype(np.float32)
    B = B_df.values.astype(np.float32)

    return A, B


# 2. 数据预处理
def preprocess_data(A, B, test_size=0.2, random_state=42):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        A, B, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


# 3. 模型训练与评估
def train_and_evaluate(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    return mse, mae, r2

# 4. 主函数
def main():
    # 加载数据
    A, B = load_data()

    # 预处理数据
    X_train, X_test, y_train, y_test = preprocess_data(A, B)

    # 定义不同模型
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression (L2)': Ridge(alpha=1.0),
        'Lasso Regression (L1)': Lasso(alpha=0.1),
        'Linear Regression with Scaling': make_pipeline(StandardScaler(), LinearRegression()),
        'Ridge Regression with Scaling': make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    }

    results = {}
    for name, model in models.items():
        mse, mae, r2 = train_and_evaluate(X_train, X_test, y_train, y_test, model)
        results[name] = (mse, mae, r2)

        print(f"====== {name} 拟合结果 ======")
        print(f"Bias MSE: {mse:.4f}")
        print(f"Bias MAE: {mae:.4f}")
        print(f"Bias R^2: {r2:.4f}\n")

    return results


if __name__ == "__main__":
    results = main()

#最小二乘+SVD（带有偏置项）
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import os

#1.读取数据
folder='./Data/附件1(Attachment 1)'
A_path=os.path.join(folder,'A.xlsx')
B_path=os.path.join(folder,'B.xlsx')

A_df=pd.read_excel(A_path,header=None)#第一行默认是列名,没有被计入
B_df=pd.read_excel(B_path,header=None)

A=A_df.values.astype(np.float32)
B=B_df.values.astype(np.float32)




# 1. 为 A 增加一列全是 1 的偏置项 (10000, 100) => A_bias
A_augmented = np.hstack([A, np.ones((A.shape[0], 1), dtype=np.float32)])  # (10000, 100)

# 2. 重新进行 SVD 拟合
U, S, VT = np.linalg.svd(A_augmented, full_matrices=False)
S_inv = np.diag(1. / S)
w_aug = VT.T @ S_inv @ U.T @ B

# 3. 预测
B_pred_bias = A_augmented @ w_aug

# 4. 评估
mse_bias = mean_squared_error(B, B_pred_bias)
mae_bias = mean_absolute_error(B, B_pred_bias)
r2_bias = r2_score(B, B_pred_bias)

print("\n====== 最小二乘法 + SVD 拟合结果 ======")
print(f"Bias MSE: {mse_bias:.4f}")
print(f"Bias MAE: {mae_bias:.4f}")
print(f"Bias R^2: {r2_bias:.4f}")