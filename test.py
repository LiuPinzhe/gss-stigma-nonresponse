import pandas as pd

# 关键：关闭分类转换
df = pd.read_stata("gss7224_r1.dta", convert_categoricals=False)
print(df.shape)
print(df.head())
