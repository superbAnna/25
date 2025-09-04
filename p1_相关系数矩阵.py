import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

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

# 选取相关列
cols = ['Y染色体浓度', '检测孕周', '孕妇BMI', '年龄']
data = df[cols].dropna()

# 计算相关性
corr = data.corr()
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Y染色体浓度与孕周、BMI、年龄相关性热力图')
plt.show()

sns.pairplot(data)
plt.show()