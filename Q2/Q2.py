import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

# --------------------- 数据加载 ---------------------
# 加载数据（假设数据为CSV文件，数据路径可根据实际情况调整）
df = pd.read_csv("Finaldata.csv")

# 显示数据前几行，检查数据加载是否正确
print(df.head())

# --------------------- 数据预处理 ---------------------

# 假设我们关心的数据列有：
# - Year: 年份
# - Country: 国家
# - Host: 是否为东道主
# - Gold, Silver, Bronze: 奖牌数量
# - Total: 总奖牌数
# - Gold_Ratio: 金牌比率
# - RPI, CMR, MCI, ECR, ME 等：其他特征

# 创建奖牌总数的列，如果没有的话
df['Total'] = df['Gold'] + df['Silver'] + df['Bronze']

# 计算奖牌数变化，假设我们使用"Total"列作为因变量
df['Total_diff'] = df.groupby('Country')['Total'].diff().fillna(0)

# --------------------- 双重差分模型 ---------------------

# 双重差分模型：假设Treat（处理组：东道主）和Post（时间变量：年份2012之后）影响奖牌数
df['Post'] = df['Year'] >= 2012  # 2012年后为“Post”组
df['Treat'] = df['Host'].apply(lambda x: 1 if x == 'Greece' else 0)  # 假设“Greece”为处理组，其他为对照组
df['Treat_Post'] = df['Treat'] * df['Post']  # 双重差分项

# 使用OLS回归进行分析
model = ols("Total_diff ~ Treat + Post + Treat_Post + Host", data=df).fit()

# 输出回归结果
print("\n双重差分模型回归结果：")
print(model.summary())

# --------------------- 可视化：处理前后奖牌数变化 ---------------------

# 计算每年和处理组的平均奖牌数
avg_medals = df.groupby(["Year", "Treat"])["Total"].mean().reset_index()

# 绘制奖牌数变化趋势图
plt.figure(figsize=(8, 6))
for treat, group in avg_medals.groupby("Treat"):
    plt.plot(group["Year"], group["Total"], label=f"Treat={treat}")
plt.axvline(x=2012, color="red", linestyle="--", label="2012年后")
plt.title("Medal Change")
plt.xlabel("Year")
plt.ylabel("Average Total Medals")
plt.legend()
plt.show()

# --------------------- 奖牌效率（ME）和项目覆盖率（ECR）建模 ---------------------

# 项目覆盖率（ECR）计算函数
def calculate_ecr(M, P, delta_C, NR, C_t):
    return (M / P) * np.log(1 + delta_C * C_t * NR)

# 奖牌效率（ME）计算函数
def calculate_me(BaseME, A, theta_C, C_t):
    return (BaseME / A) * (1 + theta_C * C_t)

# 假设我们选择某些国家的数据进行计算
example_country = df[df['Country'] == 'Australia'].iloc[0]

# 提取相关的特征
M = example_country['Total']  # 奖牌数
P = example_country['M']  # 项目数
delta_C = 0.25  # 教练效应对项目覆盖率提升的边际贡献
NR = example_country['predicted_NR']  # 资源配置能力（根据数据列）
C_t = example_country['Predicted_CA']  # 教练效应（根据数据列）

# 计算项目覆盖率（ECR）和奖牌效率（ME）
ECR = calculate_ecr(M, P, delta_C, NR, C_t)
BaseME = 0.6  # 基础奖牌效率
A = example_country['A']  # 资源投入
ME = calculate_me(BaseME, A, 0.15, C_t)

print(f"\nAustralia 的项目覆盖率 (ECR): {ECR}")
print(f"Australia 的奖牌效率 (ME): {ME}")
# --------------------- 处理缺失值 ---------------------
# 先检查关键列是否存在缺失值
required_columns = ['ECR', 'ME', 'NR', 'Degree Centrality']
missing_data = df[required_columns].isnull().sum()

# 输出缺失值数量
print("\n每个列的缺失值数量：")
print(missing_data)

# 填充缺失值（假设用0填充，可以根据实际情况选择填充策略）
df[required_columns] = df[required_columns].fillna(0)
# --------------------- 选择适合投资“优秀”教练的国家 ---------------------

# 示例国家数据
countries = {
    "Australia": {"ECR": ECR, "ME": ME, "NR": NR, "DegreeCentrality": example_country['Degree Centrality']},
    "USA": {"ECR": 0.6, "ME": 0.5, "NR": 0.4, "DegreeCentrality": 0.35},
    "Germany": {"ECR": 0.5, "ME": 0.7, "NR": 0.3, "DegreeCentrality": 0.4},
}

# 投资分析：计算每个国家的奖牌提升预测
print("\n每个国家投资优秀教练后的奖牌提升预测：")
for country, stats in countries.items():
    # 假设我们选择体系改造型教练进行投资
    projected_medals_increase = (stats["ECR"] * delta_C + stats["ME"] * 0.1) * 5
    print(f"{country} - 预计奖牌提升: {projected_medals_increase:.2f} 枚奖牌")
