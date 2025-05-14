import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv("Finaldata.csv")

# 假设 T_c 是教练的国际影响力，S_c 是教练的体系改造能力
# 设定教练效应的权重
gamma_1 = 0.7  # 国际影响力
gamma_2 = 0.3  # 体系改造能力

# ---------------------- 计算教练效应 ----------------------

# 计算教练的国际影响力 T_c（基于历史奖牌数的平均）
df['T_c'] = df.groupby('NOC')['Total'].transform('mean')

# 计算教练的体系改造能力 S_c（基于 ECR, MCI, ME）
df['S_c'] = df['ECR'] * (1 - df['MCI']) + df['ME']

# 计算伟大教练效应 C_t
df['Coach_Effect'] = gamma_1 * df['T_c'] + gamma_2 * df['S_c']

# ---------------------- 项目覆盖率（ECR） ----------------------

# 假设 `M` 为每个国家参加的奖牌数，`P_j` 为项目总数（从 `Total_Projects` 列中获取）
# 假设 `Predicted_NR` 为国家的资源配置能力
delta_C = 0.1  # 教练效应对项目覆盖率的边际贡献

# 计算项目覆盖率 ECR
df['ECR'] = df['ECR']* np.log(1 + delta_C * df['Coach_Effect'] * df['predicted_NR'])

# ---------------------- 奖牌效率（ME） ----------------------

# 假设 `A` 为运动员数（这个值应该在数据中存在，如果没有则需要计算）
theta_C = 0.1  # 教练效应对奖牌效率的边际贡献

# 计算奖牌效率 ME
df['ME'] = df['ME']*(1 + theta_C * df['Coach_Effect'])

# ---------------------- 确保每个国家每年只有一行数据 ----------------------

# # 聚合每年每个国家的所有计算结果，确保每个国家每年只有一行数据
# df_aggregated = df.groupby(['Year', 'NOC'])

# ---------------------- 保存结果 ----------------------

df=df.drop_duplicates()

# 保存计算后的数据到新的 CSV 文件
df.to_csv("Q2data.csv", index=False)

# 打印结果查看
print(df[['Year', 'NOC', 'Total', 'ECR', 'ME', 'Coach_Effect']].head())
