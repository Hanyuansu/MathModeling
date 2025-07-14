import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import  StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

#1.加载数据
X_path='./Data/附件3(Attachment 3)/3-X.xlsx'
Y_path='./Data/附件3(Attachment 3)/3-Y.xlsx'

X=pd.read_excel(X_path,header=None).values.astype(np.float32)
Y=pd.read_excel(Y_path,header=None).values.flatten().astype(np.float32)

print(f"X形状:{X.shape},Y形状:{Y.shape}")

#2.滤波法去噪(方差去噪，去掉低方差特征)
from sklearn.feature_selection import  VarianceThreshold
selector=VarianceThreshold(threshold=1e-5)
X_filtered=selector.fit_transform(X)
print(f"方差过滤后X形状:{X_filtered.shape}")

#3.PCA去噪(保留95%方差)，这里的参数应该如何选择
pca=PCA(n_components=0.95)
X_pca=pca.fit_transform(X_filtered)
print(f"PCA降维后X形状:{X_pca.shape}")

#4.特征标准化
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X_pca)

#5.建立多元线性回归模型
model=LinearRegression()
model.fit(X_scaled,Y)
Y_pred=model.predict(X_scaled)

#6.MSE评估
mse=mean_squared_error(Y,Y_pred)
print(f"\n均方误差(MSE):{mse:.6f}")

#7.F检验和t检验
#用statsmodels获取详细统计量
X_scaled_sm=sm.add_constant(X_scaled)
ols_model=sm.OLS(Y,X_scaled_sm).fit()

print("\n回归模型摘要")
print(ols_model.summary())

#F统计量(模型整体显著性)
f_stat=ols_model.fvalue
f_pvalue=ols_model.f_pvalue
print(f"\nF检验:F={f_stat:.4f},p={f_pvalue:.4e}")

