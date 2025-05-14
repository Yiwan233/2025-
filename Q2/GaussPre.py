import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import logging
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern
# 设置日志配置
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置工作目录
os.chdir(r'D:\CODE\MCM\C\Q2')
logging.info("工作目录已设置。")

# 载入数据
df = pd.read_csv('final_data1.csv')
logging.info("数据已载入。")

# 构建特征矩阵
features = ['CMR', 'RPI', 'MCI', 'ECR', 'ME', 'RIR', 'GR', 'delta_GR', 'Adjusted Degree Centrality', 'Adjusted Betweenness Centrality', 'Adjusted Closeness Centrality']
logging.debug(f"使用的特征: {features}")

# 添加滞后特征
df['Gold_Lag'] = df.groupby('NOC')['Gold'].shift(1)  # 上一年的金牌数
df['Total_Lag'] = df.groupby('NOC')['Total'].shift(1)  # 上一年的总奖牌数
logging.info("滞后特征已添加。")

# 处理缺失值
df.fillna(0, inplace=True)
logging.info("缺失值已处理。")

# 选择要预测的目标变量
target_gold = 'Gold'  # 预测金牌数
target_total = 'Total'  # 预测总奖牌数
logging.debug(f"目标变量: {target_gold}, {target_total}")

# 准备训练数据和特征
X = df[features + ['Gold_Lag', 'Total_Lag']]  # 特征包括滞后值
y_gold = df[target_gold]
y_total = df[target_total]
logging.info("训练数据和目标变量已准备好。")

# 初始化高斯回归模型

kernel = Matern(length_scale=20.0, nu=0.5)
gpr_gold = GaussianProcessRegressor(kernel=kernel, alpha=1.0, n_restarts_optimizer=30)  # 增大噪声水平
gpr_total = GaussianProcessRegressor(kernel=kernel, alpha=1.0, n_restarts_optimizer=30)

logging.info("高斯过程模型已初始化。")

# 训练高斯回归模型：金牌数
gpr_gold.fit(X, y_gold)
logging.info("金牌数模型已训练。")

# 训练高斯回归模型：总奖牌数
gpr_total.fit(X, y_total)
logging.info("总奖牌数模型已训练。")

# 遍历每个国家进行预测
result_list = []

# 放大标准差的影响
std_scale = 4 # 放大标准差的影响


for noc in df['NOC'].unique():

    # 获取该国家的最后一年的数据
    country_data = df[df['NOC'] == noc].iloc[-1:]

    
    # 获取特征数据
    X_country = country_data[features + ['Gold_Lag', 'Total_Lag']].values
    
    # 预测金牌数和总奖牌数
    prediction_gold, std_gold = gpr_gold.predict(X_country, return_std=True)
    prediction_total, std_total = gpr_total.predict(X_country, return_std=True)
    logging.debug(f"预测金牌: {prediction_gold[0]}, 标准差: {std_gold[0]}\n预测总奖牌: {prediction_total[0]}, 标准差: {std_total[0]}")
    
    # 将预测值转换为区间，并确保金牌数不为负
    gold_range = (max(0, prediction_gold[0] - std_scale * std_gold[0]), max(0, prediction_gold[0] + std_scale * std_gold[0]))
    total_range = (prediction_total[0] - std_scale * std_total[0], prediction_total[0] + std_scale * std_total[0])

    
    # 保存每个国家的预测结果
    result_list.append({
        'NOC': noc,
        'Predicted Gold Min': gold_range[0],
        'Predicted Gold Max': gold_range[1],
        'Predicted Total Min': total_range[0],
        'Predicted Total Max': total_range[1]
    })

# 将结果保存到数据框中
results_df = pd.DataFrame(result_list)
logging.info("预测结果已保存到数据框。")

# 保存到CSV文件
results_df.to_csv('predicted_medals_with_uncertainty.csv', index=False)
logging.info("预测结果已保存到CSV文件。")

# 可视化预测结果（以第一个国家为例）
plt.figure(figsize=(10,6))
plt.bar(['Gold', 'Total'], [results_df['Predicted Gold Max'].iloc[0], results_df['Predicted Total Max'].iloc[0]], 
        yerr=[results_df['Predicted Gold Max'].iloc[0] - results_df['Predicted Gold Min'].iloc[0], 
              results_df['Predicted Total Max'].iloc[0] - results_df['Predicted Total Min'].iloc[0]], 
        capsize=5, color=['gold', 'gray'])
plt.title('Predicted Medals for Next Olympics')
plt.ylabel('Predicted Number of Medals')
logging.info("可视化结果已生成。")
plt.show()
