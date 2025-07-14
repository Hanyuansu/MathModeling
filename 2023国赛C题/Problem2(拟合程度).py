import os
import pandas as pd
import numpy as np
from datetime import timedelta
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ========== GPU 配置 ==========
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
strategy = tf.distribute.MirroredStrategy()
print("Num replicas:", strategy.num_replicas_in_sync)

# ========== 全局配置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
output_dir = "Problem2_output"
os.makedirs(output_dir, exist_ok=True)

# ========== 1. 读取并合并数据 ==========
sales = pd.read_excel("附件2.xlsx", engine="openpyxl")
info  = pd.read_excel("附件1.xlsx", engine="openpyxl")
cost  = pd.read_excel("附件3.xlsx", engine="openpyxl")

sales.rename(columns={'单品编码':'商品编号','销售日期':'日期','销量(千克)':'销量'}, inplace=True)
info .rename(columns={'单品编码':'商品编号','分类名称':'品类'}, inplace=True)
cost.rename(columns={'单品编码':'商品编号','日期':'日期','批发价格(元/千克)':'成本'}, inplace=True)

sales['日期'] = pd.to_datetime(sales['日期'])
cost ['日期'] = pd.to_datetime(cost['日期'])

df = (sales
      .merge(info[['商品编号','品类']], on='商品编号', how='left')
      .merge(cost[['日期','商品编号','成本']], on=['日期','商品编号'], how='left'))
df.dropna(subset=['成本'], inplace=True)

# ========== 2. 日级聚合 ==========
daily = df.groupby(['日期','品类']).agg(
    total_sales=('销量','sum'),
    avg_cost   =('成本','mean')
).reset_index()

sales_ts = daily.pivot(index='日期', columns='品类', values='total_sales').fillna(0)
cost_ts  = daily.pivot(index='日期', columns='品类', values='avg_cost').ffill()

# ========== 3. 归一化与滑动窗口 ==========
window  = 14
horizon = 7
train_ratio = 0.9
scaler_s = {c: MinMaxScaler() for c in sales_ts.columns}
scaler_c = {c: MinMaxScaler() for c in cost_ts.columns}

eval_records = []
pred_sales = pd.DataFrame(
    index=pd.date_range(sales_ts.index[-1] + timedelta(days=1), periods=horizon, freq='D'),
    columns=sales_ts.columns
)
pred_cost = pred_sales.copy()

for cat in sales_ts.columns:
    print(f"\n==== 正在处理品类：{cat} ====")
    s = sales_ts[cat].values
    c = cost_ts[cat].values

    if len(s) < window + horizon:
        print(f"[跳过] 样本数太少 ({len(s)} 天)，至少需要 {window + horizon} 天")
        continue

    s_scaled = scaler_s[cat].fit_transform(s.reshape(-1,1)).flatten()
    c_scaled = scaler_c[cat].fit_transform(c.reshape(-1,1)).flatten()

    Xs, ys = [], []
    Xc, yc = [], []
    for i in range(len(s_scaled) - window):
        Xs.append(s_scaled[i:i+window]); ys.append(s_scaled[i+window])
        Xc.append(c_scaled[i:i+window]); yc.append(c_scaled[i+window])
    Xs = np.array(Xs).reshape(-1, window, 1); ys = np.array(ys)
    Xc = np.array(Xc).reshape(-1, window, 1); yc = np.array(yc)

    if len(Xs) == 0:
        print(f"[跳过] 无法构建样本")
        continue

    train_end = int(len(Xs) * train_ratio)
    if train_end == 0 or train_end >= len(Xs):
        print(f"[跳过] 验证集太小或不存在，Xs样本={len(Xs)}，train_end={train_end}")
        continue

    Xs_tr, ys_tr = Xs[:train_end], ys[:train_end]
    Xs_val, ys_val = Xs[train_end:], ys[train_end:]
    Xc_tr, yc_tr = Xc[:train_end], yc[:train_end]
    Xc_val, yc_val = Xc[train_end:], yc[train_end:]

    def build_and_train(X_train, y_train, X_val=None, y_val=None):
        with strategy.scope():
            model = Sequential([
                Input(shape=(window,1)),
                LSTM(6, return_sequences=False),
                Dense(7, activation='relu'),
                Dense(1, activation='linear')
            ])
            model.compile(optimizer=Adam(), loss='mse')
        callbacks = [EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                                   patience=10, restore_best_weights=True)]
        fit_kwargs = dict(x=X_train, y=y_train, epochs=100,
                          batch_size=16, verbose=1, callbacks=callbacks)
        if X_val is not None and len(X_val)>0:
            fit_kwargs['validation_data'] = (X_val, y_val)
        model.fit(**fit_kwargs)
        return model

    model_s = build_and_train(Xs_tr, ys_tr, Xs_val, ys_val)
    model_c = build_and_train(Xc_tr, yc_tr, Xc_val, yc_val)

    def eval_and_inv(model, X_val, y_val, scaler):
        if len(X_val)==0:
            return np.nan, np.nan, np.nan, np.nan
        y_pred = model.predict(X_val, verbose=0).flatten()
        y_true_o = scaler.inverse_transform(y_val.reshape(-1,1)).flatten()
        y_pred_o = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
        return (
            round(r2_score(y_true_o, y_pred_o),3),
            round(mean_squared_error(y_true_o, y_pred_o),2),
            round(np.sqrt(mean_squared_error(y_true_o, y_pred_o)),2),
            round(mean_absolute_error(y_true_o, y_pred_o),2)
        )

    r2_s, mse_s, rmse_s, mae_s = eval_and_inv(model_s, Xs_val, ys_val, scaler_s[cat])
    r2_c, mse_c, rmse_c, mae_c = eval_and_inv(model_c, Xc_val, yc_val, scaler_c[cat])

    eval_records.append({
        '品类': cat,
        'R2_销量': r2_s, 'MSE_销量': mse_s, 'RMSE_销量': rmse_s, 'MAE_销量': mae_s,
        'R2_进价': r2_c, 'MSE_进价': mse_c, 'RMSE_进价': rmse_c, 'MAE_进价': mae_c
    })

    seq_s = list(s_scaled[-window:])
    seq_c = list(c_scaled[-window:])
    for t in range(horizon):
        ps = float(model_s.predict(np.array(seq_s[-window:]).reshape(1,window,1), verbose=0))
        pc = float(model_c.predict(np.array(seq_c[-window:]).reshape(1,window,1), verbose=0))
        pred_sales.iloc[t][cat] = scaler_s[cat].inverse_transform([[ps]])[0,0]
        pred_cost .iloc[t][cat] = scaler_c[cat].inverse_transform([[pc]])[0,0]
        seq_s.append(ps); seq_c.append(pc)

# ========== 保存输出 ==========
pd.DataFrame(eval_records).to_excel(
    os.path.join(output_dir, "LSTM_验证评估_全历史_GPU.xlsx"), index=False)
pred_sales.to_excel(os.path.join(output_dir, "未来7日销量_LSTM_全历史_GPU.xlsx"))
pred_cost .to_excel(os.path.join(output_dir, "未来7日进价_LSTM_全历史_GPU.xlsx"))

print("已保存：")
print(" - LSTM_验证评估_全历史_GPU.xlsx")
print(" - 未来7日销量_LSTM_全历史_GPU.xlsx")
print(" - 未来7日进价_LSTM_全历史_GPU.xlsx")
