import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 读取数据
df = pd.read_csv('final_data.csv')

# 定义ARIMAX模型
# 这里我们假设CA是目标变量（Y），CMR, RPI, GCR是外生变量（X）

# 选择外生变量 CMR, RPI, GCR
X = df[['CMR', 'RPI', 'GCR','EI','ESP','ECR_Large','ECR_Small']]  # 外生变量
X = sm.add_constant(X)  # 添加常数项（截距）

# 目标变量（CA）
y = df['Total']  # 目标变量（奖牌数，作为CA的代理）

# 划分训练集和测试集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 拟合ARIMAX模型，设定p=1, d=1, q=1为示例（可以根据AIC/BIC等选择更合适的p, d, q）
model = ARIMA(y_train, exog=X_train, order=(1, 1, 1))

# 拟合模型
result = model.fit()

# 输出模型结果
print(result.summary())

# 使用训练好的模型预测每个国家的CA值
forecast_steps = len(X_test)  # 用测试集的长度作为预测步数

# 使用模型生成预测值
forecast = result.forecast(steps=forecast_steps, exog=X_test)

# 将预测结果存入DataFrame
forecast_df = pd.DataFrame({
    'Country': df['Country'].iloc[-forecast_steps:],  # 获取预测步数的国家名称
    'Predicted_CA': forecast  # 存储预测的CA值
})

# 显示预测结果
print(forecast_df)

# 绘制实际数据和预测数据的折线图
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], y, label='Actual CA', color='blue', marker='o')
plt.plot(df['Year'][-forecast_steps:], forecast, label='Forecasted CA', color='red', marker='x')
plt.title('Actual vs Forecasted CA (Cumulative Advantage)')
plt.xlabel('Year')
plt.ylabel('Total CA')
plt.legend()
plt.grid(True)
plt.savefig('ARIMAX_Forecast_CA.png')
plt.show()

# 保存预测结果为 CSV 文件
forecast_df.to_csv('CA.csv', index=False)
