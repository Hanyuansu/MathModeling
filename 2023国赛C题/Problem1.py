import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.dates as mdates
from scipy.signal import find_peaks
import os

# 设置中文显示（需系统支持中文字体）
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 创建统一输出文件夹
output_dir = "Problem1 output"
os.makedirs(output_dir, exist_ok=True)
peak_dir = os.path.abspath(os.path.join(output_dir, "周期高峰图"))
os.makedirs(peak_dir, exist_ok=True)

# 图像保存函数
def save_and_show(fig, filename):
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)


# 1. 读取数据
sales_df = pd.read_excel("附件2.xlsx")
info_df = pd.read_excel("附件1.xlsx")

# 2. 重命名 & 合并品类
sales_df.rename(columns={'单品编码': '商品编号', '销量(千克)': '销售数量'}, inplace=True)
info_df.rename(columns={'单品编码': '商品编号', '分类名称': '品类名称'}, inplace=True)

sales_df['销售日期'] = pd.to_datetime(sales_df['销售日期'])
sales_df = sales_df[['销售日期', '商品编号', '销售数量']].dropna()
merged_df = pd.merge(sales_df, info_df[['商品编号', '品类名称']], on='商品编号', how='left')
merged_df.to_excel(os.path.join(output_dir, "合并后销售数据.xlsx"), index=False)

# 3. 生成每日品类销量表
daily_sales = merged_df.groupby(['销售日期', '品类名称'])['销售数量'].sum().unstack().fillna(0)
daily_sales.to_excel(os.path.join(output_dir, "每日品类销量.xlsx"))

# 4. 日销量趋势图
fig, ax = plt.subplots(figsize=(14, 6))
for col in daily_sales.columns:
    ax.plot(daily_sales.index, daily_sales[col], label=col)
ax.set_title("每日各蔬菜品类销售趋势图")
ax.set_xlabel("日期")
ax.set_ylabel("销量 (kg)")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()
plt.show()
save_and_show(fig, "每日销量趋势图.png")

# 5. 箱线图
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=daily_sales, ax=ax)
ax.set_title("各品类日销量分布（箱线图）")
ax.set_ylabel("销量 (kg)")
fig.autofmt_xdate()
save_and_show(fig, "日销量箱线图.png")

# 6. 周期性分析 - 周均值
daily = merged_df.groupby(['销售日期', '品类名称'])['销售数量'].sum().reset_index()
daily['星期'] = daily['销售日期'].dt.dayofweek
week_avg = daily.groupby(['星期', '品类名称'])['销售数量'].mean().unstack()
week_avg.to_excel(os.path.join(output_dir, "周平均销量.xlsx"))

fig, ax = plt.subplots(figsize=(12, 6))
week_avg.plot(kind='bar', ax=ax)
ax.set_title("各蔬菜品类按星期的平均销量")
ax.set_ylabel("平均销量 (kg)")
save_and_show(fig, "周周期性趋势图.png")

# 7. 月季节性分析
daily = merged_df.groupby(['销售日期', '品类名称'])['销售数量'].sum().reset_index()
daily['月份'] = daily['销售日期'].dt.month
month_avg = daily.groupby(['月份', '品类名称'])['销售数量'].mean().unstack()
month_avg.to_excel(os.path.join(output_dir, "月度平均销量.xlsx"))

fig, ax = plt.subplots(figsize=(12, 6))
month_avg.plot(ax=ax)
ax.set_title("各蔬菜品类季节性销售趋势（按月份）")
ax.set_xlabel("月份")
ax.set_ylabel("平均销量 (kg)")
save_and_show(fig, "月度季节性趋势图.png")

# 8. 相关性分析（皮尔逊）
corr_matrix = daily_sales.corr()
corr_matrix.to_excel(os.path.join(output_dir, "销量皮尔逊相关性矩阵.xlsx"))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title("品类销量相关性热力图（Pearson）")
save_and_show(fig, "品类相关性热力图.png")

# 9. 正态性检验（QQ图 + P值）
p_results = []
all_categories = daily_sales.columns
ncols = 3
nrows = (len(all_categories) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()
i=-1

for i, category in enumerate(all_categories):
    stats.probplot(daily_sales[category], dist="norm", plot=axes[i])
    axes[i].set_title(f"{category}")
    stat, p_val = stats.shapiro(daily_sales[category])
    p_results.append({'品类': category, 'P值': p_val})

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("QQ图：检验销量是否正态分布", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(os.path.join(output_dir, "QQ图拼图.png"), dpi=300, bbox_inches='tight')
plt.close(fig)

p_df = pd.DataFrame(p_results)
p_df['是否正态'] = p_df['P值'].apply(lambda x: '是' if x > 0.05 else '否')
p_df.to_excel(os.path.join(output_dir, "各品类销量正态性检验结果.xlsx"), index=False)

# 10. 周期性高峰检测（每周一次）
for category in daily_sales.columns:
    series = daily_sales[category]
    peaks, _ = find_peaks(series, distance=7)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(series, label='销量', linewidth=1.2)
    ax.plot(series.index[peaks], series.iloc[peaks], "x", color='red', label='高峰')
    ax.set_title(f"{category}销量周期性高峰（每周）")
    ax.set_xlabel("日期")
    ax.set_ylabel("销量 (kg)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    fig.savefig(os.path.join(peak_dir, f"{category}_高峰图.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
