import pandas as pd
import numpy as np
import re
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

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

# 预测与输出
pred = svr.predict(X_poly)
print('SVR模型拟合完成，前10个预测值：')
print(pred[:10])
