import pandas as pd
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

# 读取清洗后的数据
df = pd.read_csv("final_data.csv")


# 查看数据中是否有缺失值
print("缺失值检查:")
print(df.isnull().sum())

# 处理缺失值（这里假设空值填充为0，你可以根据需要调整策略）
df.fillna(0, inplace=True)

# 设置数据为面板数据格式：以 'NOC' 和 'Year' 为索引
df = df.set_index(['NOC', 'Year'])

# 定义自变量和因变量
# 因变量NR的目标是通过模型计算得出，因此不直接定义NR
X = df[['MCI', 'ECR', 'ME', 'RIR']]  # 自变量包括 MCI、ECR、ME 和 RIR

# 添加常数项（截距）
X = sm.add_constant(X)

# 固定效应模型：使用 PanelOLS 模型，并添加国家固定效应和时间固定效应
y = df['Total']  # 如果你想预测奖牌总数（Total）作为因变量
model_fe = PanelOLS(y, X, entity_effects=True, time_effects=True)
fe_results = model_fe.fit()

# 输出固定效应模型的结果
print(fe_results.summary)

# 计算每个国家每一届奥运会的资源指数 NR
# 通过模型系数计算 NR：β0 + β1*MCI + β2*ECR + β3*ME + β4*RIR
df['predicted_NR'] = fe_results.params['const'] + \
                      fe_results.params['MCI'] * df['MCI'] + \
                      fe_results.params['ECR'] * df['ECR'] + \
                      fe_results.params['ME'] * df['ME'] + \
                      fe_results.params['RIR'] * df['RIR']

# 获取国家和年份的列，以便最终按时间排序并保存
df_reset = df.reset_index()

# 保存计算结果到 CSV 文件
df_reset.to_csv('NR.csv', index=False)

print("\n计算完成，结果已保存到 'calculated_NR.csv' 文件。")