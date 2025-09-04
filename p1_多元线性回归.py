import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import re 
import numpy as np
# 读取数据
df = pd.read_excel('附件.xlsx')
# 定义孕周转换函数，如'11w+6'转为11.86
def parse_week(week_str):
	match = re.match(r'(\d+)w\+(\d+)', str(week_str))
	if match:
		week = int(match.group(1))
		day = int(match.group(2))
		return week + day / 7
	try:
		return float(week_str)
	except:
		return None

# 转换检测孕周为数值型
df['检测孕周'] = df['检测孕周'].apply(parse_week)
# 只保留相关列且去除无效值
cols = ['检测孕周', '孕妇BMI', '年龄', 'Y染色体浓度']
data = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

X = data[['检测孕周', '孕妇BMI']]

X = sm.add_constant(X)  # 加截距项
# 因变量
y = data['Y染色体浓度']

# 建立回归模型
model = sm.OLS(y, X).fit()
print(model.summary())

# 绘制散点图