import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

##Python实现PCA
def pca(X,k):#k is the components you want
  #mean of each feature
  n_samples, n_features = X.shape
  mean=X.mean(axis=0)
  variance=X.var(axis=0)
  #normalization
  norm_X=(X-mean)/np.sqrt(variance)
  #scatter matrix
  scatter_matrix=np.dot(np.transpose(norm_X),norm_X)/(n_samples-1)
  #Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(reverse=True)
  # select the top k eig_vec
  feature=np.array([ele[1] for ele in eig_pairs[:k]])
  #get new data
  data=np.dot(norm_X,np.transpose(feature))
  return data,eig_val,eig_vec,mean,variance

df=pd.read_excel("data.xlsx")
X=df.iloc[:,1:]
X=np.array(X)
city = df.iloc[:, 0].values          # 城市名
feature_names = df.columns[1:]      # 指标名

# 自己写的 PCA
X_pca, eig_val, eig_vec, mean, var = pca(X, 2)

print("主成分得分：")
print(X_pca)

# 方差贡献率
explained_ratio = eig_val / eig_val.sum()
print("方差贡献率：", explained_ratio)
print("累计方差贡献率：", np.cumsum(explained_ratio))

# 城市在 PC1, PC2 上的得分表
scores_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
scores_df.insert(0, "城市", city)
print(scores_df.head())

# 载荷矩阵（每个原始变量在 PC1, PC2 上的系数）
loadings = eig_vec[:, :2] * np.sqrt(eig_val[:2])   # 想简单点也可以直接用 eig_vec[:, :2]
loadings_df = pd.DataFrame(loadings,
                           index=feature_names,
                           columns=["PC1", "PC2"])
print("载荷矩阵：")
print(loadings_df)


# print(X)
# MEAN=X.mean(axis=0)
# #print(MEAN)
# Variance=X.var(axis=0)
# # X_p=(X-MEAN)/np.sqrt(Variance)
# # print(X_p)
# scaler=StandardScaler()
# X_scaler=scaler.fit_transform(X)
# print(X_scaler)
# COV=np.dot(X_scaler.T,X_scaler)
# W,V=np.linalg.eig(COV)
# sum_W=np.sum(W)
# print(sum_W)
# f=np.divide(W,sum_W)
# print(f)

# X_pca=pca(X,2)
# print(X_pca)
# pca=PCA(n_components=2)
# X_PCA=pca.fit_transform(X)
# print(X_PCA)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_)

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca_sklearn = PCA(n_components=2)
Z_sklearn = pca_sklearn.fit_transform(X_std)

print("\n自己实现的 Z（前 5 行）：")
print(X_pca[:5])
print("\nsklearn 的 Z（前 5 行）：")
print(Z_sklearn[:5])