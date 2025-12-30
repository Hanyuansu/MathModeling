import numpy as np
from scipy.io import loadmat

# 读入 .mat 文件
mat = loadmat("irisData.mat")
A = mat["A"]
print(A)

# 前四列是特征，最后一列是类别标签（1,2,3）
X = A[:, :4]
y = A[:, 4].astype(int)


def lda(X, y, k):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    cls = np.unique(y)
    n_samples, n_features = X.shape

    mu = X.mean(axis=0)              # 全局均值
    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))

    for c in cls:
        Xc = X[y == c]
        mu_c = Xc.mean(axis=0)
        xc = Xc - mu_c
        Sw += xc.T @ xc

        diff = (mu_c - mu).reshape(-1, 1)
        Sb += Xc.shape[0] * (diff @ diff.T)

    M = np.linalg.pinv(Sw) @ Sb
    eigvals, eigvecs = np.linalg.eig(M)
    idx = np.argsort(-eigvals.real)
    W = eigvecs[:, idx[:k]].real     # 投影矩阵

    return W, mu

# 将 4 维特征投影到 2 维
W, mu = lda(X, y, k=2)
X_lda = (X - mu) @ W    # X_lda 形状为 (150, 2)

import matplotlib.pyplot as plt

plt.figure()
for c, m in zip([1, 2, 3], ["o", "s", "^"]):
    plt.scatter(X_lda[y == c, 0], X_lda[y == c, 1], marker=m, label=f"class {c}")
plt.legend()
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("LDA on Iris")
plt.show()
#
#
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 读取鸢尾花数据
mat = loadmat("irisData.mat")
A = mat["A"]

# 特征与标签
X = A[:, :4]
y = A[:, 4].astype(int)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.04, random_state=0, stratify=y
)

# 线性判别分析分类器
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# 在测试集上预测
y_pred = lda.predict(X_test)

# 分类效果评估
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("测试集准确率:", acc)
print("混淆矩阵:")
print(cm)
print("分类报告:")
print(classification_report(y_test, y_pred))

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

mat = loadmat("irisData.mat")
A = mat["A"]

X = A[:, :4].astype(float)
y = A[:, 4].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.04, random_state=0, stratify=y
)

def lda_train(X, y):
    classes = np.unique(y)
    n, d = X.shape
    mu = {}
    pi = {}
    Sigma = np.zeros((d, d))
    for c in classes:
        Xc = X[y == c]
        mu[c] = Xc.mean(axis=0)
        pi[c] = Xc.shape[0] / n
        xc = Xc - mu[c]
        Sigma += xc.T @ xc
    Sigma /= (n - len(classes))
    return classes, mu, pi, Sigma

def lda_predict(X, classes, mu, pi, Sigma):
    inv_Sigma = np.linalg.inv(Sigma)
    n = X.shape[0]
    K = len(classes)
    scores = np.zeros((n, K))
    for j, c in enumerate(classes):
        m = mu[c]
        w = inv_Sigma @ m
        w0 = -0.5 * m.T @ inv_Sigma @ m + np.log(pi[c])
        scores[:, j] = X @ w + w0
    idx = np.argmax(scores, axis=1)
    return classes[idx]

classes, mu, pi, Sigma = lda_train(X_train, y_train)
y_pred = lda_predict(X_test, classes, mu, pi, Sigma)

acc = (y_pred == y_test).mean()
cm = confusion_matrix(y_test, y_pred)

print("测试集准确率:", acc)
print("混淆矩阵:")
print(cm)
