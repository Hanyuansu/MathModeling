import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. 读取鸢尾花数据
mat = loadmat("irisData.mat")
A = mat["A"]

# 特征矩阵 X 和 标签向量 y
X = A[:, :4]
y = A[:, 4].astype(int)  # 转成整数类型标签

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.04, random_state=0, stratify=y
)

# 3. 特征标准化
# 神经网络对特征尺度比较敏感，如果不标准化，训练会变慢或不收敛
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)  # 用训练集拟合标准化参数，并对训练集变换
X_test_std = scaler.transform(X_test)        # 用同一组标准化参数变换测试集

# 4. 构建并训练 ANN 模型
# 这里使用一个简单的多层感知机（MLP）：
# - hidden_layer_sizes=(8, 8)：两层隐藏层，每层 8 个神经元
# - activation='relu'：使用 ReLU 激活函数
# - solver='adam'：Adam 优化器，适合中小规模数据
# - alpha：L2 正则化系数（权重衰减），防止过拟合
# - max_iter：最大迭代次数
ann = MLPClassifier(
    hidden_layer_sizes=(8, 8),
    activation='relu',
    solver='adam',
    alpha=1e-3,
    max_iter=1000,
    random_state=0
)

# 使用标准化后的训练集进行训练
ann.fit(X_train_std, y_train)

# 5. 在测试集上进行预测
y_pred = ann.predict(X_test_std)

# 6. 分类效果评估
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("测试集准确率:", acc)
print("混淆矩阵:")
print(cm)
print("分类报告:")
print(classification_report(y_test, y_pred))
print("最终训练损失（loss）:", ann.loss_)
