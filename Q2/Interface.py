import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# 读取清洗后的数据
df = pd.read_csv('final_merged_data_filled.csv')

# 设置数据为面板数据格式：以 'NOC' 和 'Year' 为索引
df = df.set_index(['NOC', 'Year'])

# 定义因变量和自变量
y = df['Total']  # 目标变量是总奖牌数
X = df[['MCI', 'ECR', 'ME', 'RIR','EI','ESP','ECR_Large','ECR_Small']]  # 外生变量包括 MCI、ECR、ME 和 RIR

# 添加常数项（截距）
X = sm.add_constant(X)

# 固定效应模型：使用 PanelOLS 模型，并添加国家固定效应
model_fe = PanelOLS(y, X, entity_effects=True, time_effects=True)
fe_results = model_fe.fit()

# 输出固定效应模型的结果
print(fe_results.summary)  # 直接打印模型摘要

# 将结果保存到文件中
with open('fixed_effects_summary.txt', 'w') as f:
    f.write(str(fe_results.summary))
