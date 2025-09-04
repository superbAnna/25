import pandas as pd
import numpy as np
import re
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel('附件.xlsx')

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

df['检测孕周'] = df['检测孕周'].apply(parse_week)
cols = ['检测孕周', '孕妇BMI', 'Y染色体浓度']
data = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

# 构造交互项和多项式特征（如2阶，包括交互项）
X = data[['检测孕周', '孕妇BMI']]
y = data['Y染色体浓度']
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# 标准化+SVR建模
svr = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
svr.fit(X_poly, y)
pred = svr.predict(X_poly)

# 输出评价指标
r2 = r2_score(y, pred)
mse = mean_squared_error(y, pred)
print(f'SVR模型R²: {r2:.3f}')
print(f'SVR模型MSE: {mse:.3f}')

# 可视化拟合效果
plt.figure(figsize=(8,5))
plt.scatter(y, pred, alpha=0.7)
plt.xlabel('实际Y染色体浓度')
plt.ylabel('SVR预测值')
plt.title('SVR拟合效果散点图')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()

# 结果分析解释
print('结果分析：')
print('1. R²值越接近1，说明模型拟合效果越好。')
print('2. MSE越小，说明预测误差越低。')
print('3. 散点图中点越接近对角线，说明预测越准确。')
print('4. SVR可捕捉孕周、BMI及其交互项对Y染色体浓度的非线性影响。')
