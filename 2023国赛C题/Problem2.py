import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime

# ========== 0. 配置 ==========
# 解决 Matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
output_dir = "Problem2_output"
os.makedirs(output_dir, exist_ok=True)

# ========== 1. 读取与预处理 ==========
sales_df = pd.read_excel("附件2.xlsx", engine="openpyxl")
info_df  = pd.read_excel("附件1.csv", engine="openpyxl")
cost_df  = pd.read_excel("附件3.xlsx", engine="openpyxl")
loss_df  = pd.read_excel("附件4.xlsx", engine="openpyxl")

# 重命名
sales_df.rename(columns={
    '单品编码':'商品编号','销售单价(元/千克)':'售价','销量(千克)':'销量'
}, inplace=True)
info_df.rename(columns={'单品编码':'商品编号','分类名称':'品类'}, inplace=True)
cost_df.rename(columns={
    '单品编码':'商品编号','批发价格(元/千克)':'成本','日期':'销售日期'
}, inplace=True)
loss_df.rename(columns={'单品编码':'商品编号','损耗率(%)':'损耗率'}, inplace=True)

# 转日期
sales_df['销售日期'] = pd.to_datetime(sales_df['销售日期'])
cost_df ['销售日期'] = pd.to_datetime(cost_df['销售日期'])

# 合并为品类级记录
df = (sales_df
      .merge(cost_df[['销售日期','商品编号','成本']], on=['销售日期','商品编号'], how='left')
      .merge(info_df[['商品编号','品类']], on='商品编号', how='left')
      .merge(loss_df[['商品编号','损耗率']], on='商品编号', how='left')
     )
df.dropna(subset=['成本'], inplace=True)

# 计算加成率并剔除离群
df['加成率'] = (df['售价'] - df['成本']) / df['成本']
df = df[(df['加成率'] >= -0.2) & (df['加成率'] <= 2)]

# ========== 2. 多项式回归拟合并保存可视化 ==========
r2_records = []
plt.figure(figsize=(12, 8))

for i, (cat, sub) in enumerate(df.groupby('品类'), 1):
    X = sub[['加成率']].values
    y = sub['销量'].values
    pf = PolynomialFeatures(degree=2, include_bias=False)
    Xp = pf.fit_transform(X)
    lr = LinearRegression().fit(Xp, y)
    y_pred = lr.predict(Xp)
    r2 = r2_score(y, y_pred)
    r2_records.append({'品类': cat, 'R²': round(r2, 4)})

    plt.subplot(2, 3, i)
    plt.scatter(X, y, s=10, alpha=0.5)
    xs = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    plt.plot(xs, lr.predict(pf.transform(xs)), 'r-', linewidth=2)
    plt.title(f"{cat} (R²={r2:.3f})")
    plt.xlabel("加成率")
    plt.ylabel("销量")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "多项式回归拟合程度.png"))
plt.close()

# 保存 R² 表格
r2_df = pd.DataFrame(r2_records)
r2_df.to_excel(os.path.join(output_dir, "各品类多项式回归R2.xlsx"), index=False)
print("已保存：各品类多项式回归R2.xlsx")
print(r2_df)

# ========== 3. 构造每日销量序列并归一化 ==========
daily = (df.groupby(['销售日期','品类'])['销量']
           .sum().unstack(fill_value=0)
           .sort_index())
scalers = {}
for cat in daily.columns:
    scaler = MinMaxScaler()
    daily[cat] = scaler.fit_transform(daily[[cat]])
    scalers[cat] = scaler

# ========== 4. 随机森林+LSTM 预测未来7日销量 ==========
window = 14
future_dates = pd.date_range("2023-07-01", periods=7, freq='D')
future_preds = pd.DataFrame(index=future_dates, columns=daily.columns)

for cat in daily.columns:
    series = daily[cat].values
    # RF 特征
    X_rf = series[:-1].reshape(-1,1)
    y_rf = series[1:]
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X_rf, y_rf)
    feat = rf.predict(X_rf)

    # 拼接
    seq = np.column_stack([series[:-1], feat])

    # 滑窗
    X_lstm, y_lstm = [], []
    for j in range(len(seq) - window):
        X_lstm.append(seq[j:j+window])
        y_lstm.append(seq[j+window, 0])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    # LSTM
    model = Sequential([
        LSTM(50, input_shape=(window, 2)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_lstm, y_lstm, epochs=30, batch_size=8, verbose=0)

    # 滚动预测
    last = seq[-window:].copy()
    preds = []
    for _ in range(7):
        p = float(model.predict(last[np.newaxis], verbose=0))
        preds.append(p)
        f = rf.predict([[p]])[0]
        last = np.vstack([last[1:], [p, f]])

    # 反归一化
    future_preds[cat] = scalers[cat].inverse_transform(np.array(preds).reshape(-1,1)).flatten()

# 保存并显示预测
future_preds.to_excel(os.path.join(output_dir, "未来7日销量预测_RF_LSTM.xlsx"))
print("已保存：未来7日销量预测_RF_LSTM.xlsx")
print(future_preds)
