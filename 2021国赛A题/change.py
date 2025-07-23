import pandas as pd

# 1. 读取 CSV（encoding 视文件而定；常见有 'utf-8'、'gbk'、'gb2312'）
csv_path  = "附件3.csv"
df = pd.read_csv(csv_path, encoding='gbk')  # 如果出现乱码可改编码

# 2. 保存为 Excel：index=False 表示不要把行号写进去
excel_path = "附件3.xlsx"
df.to_excel(excel_path, index=False)

print(f"已成功转换为 {excel_path}")