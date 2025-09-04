import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

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

# 多项式特征扩展（如2阶）
X = data[['检测孕周', '孕妇BMI']]
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly = sm.add_constant(X_poly)
y = data['Y染色体浓度']

model = sm.OLS(y, X_poly).fit()
print(model.summary())
