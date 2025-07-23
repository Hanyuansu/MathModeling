import pandas as pd

# ========= 1. 文件路径 =========
file1 = "附件1.xlsx"        # 主索节点基本信息
file2 = "附件2.xlsx"        # 促动器及节点对应关系
out   = "merged.xlsx"      # 合并结果

# ========= 2. 读取 Excel =========
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# ========= 3. 统一关键列名 =========
# 假设两表关键字段都叫“主索节点编号”；若不同，可在此处重命名
key_col = "主索节点编号"
rename_dict = {
    "节点编号": key_col,          # 举例：把“节点编号”改成统一名称
    "对应主索节点编号": key_col,
}
df1.rename(columns=rename_dict, inplace=True)
df2.rename(columns=rename_dict, inplace=True)

# ========= 4. 去除重复 & 设置索引（可选） =========
# 有时同一节点会在表中出现多行，这里保留第一条并发出警告
for name, df in [("附件1", df1), ("附件2", df2)]:
    dup = df.duplicated(subset=key_col, keep=False)
    if dup.any():
        print(f"[警告] {name} 中存在重复编号：{df.loc[dup, key_col].unique()}")
        df.drop_duplicates(subset=key_col, keep="first", inplace=True)

# ========= 5. 合并 =========
merged = pd.merge(
    df1, df2,
    on=key_col,            # 连接键
    how="inner",           # 仅保留两表都有的节点；可改为 'outer' 完整并集
    suffixes=("_节点", "_促动器")  # 避免重名列冲突
)

print(f"成功合并 {len(merged)} 条记录")

# ========= 6. 保存为新的 Excel =========
merged.to_excel(out, index=False)
print(f"已保存到 {out}")
