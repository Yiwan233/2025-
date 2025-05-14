import pandas as pd
import networkx as nx
import numpy as np

# 读取清洗后的数据
medal_counts_hosts_df = pd.read_csv("testdata.csv")

# 创建一个有向图
G = nx.DiGraph()

# 假设教练效应和资源配置能力的边际贡献系数
kappa_C = 0.1  # 假设的度中心性教练效应对边际贡献
mu_C = 0.1  # 假设的介数中心性教练效应对边际贡献
nu_C = 0.1  # 假设的接近中心性教练效应对边际贡献

# 1. 按年份遍历数据，计算每年每个国家的中心性
centrality_results = []

for year in medal_counts_hosts_df['Year'].unique():
    # 每年重新构建图
    G.clear()

    # 获取当前年份的数据
    year_data = medal_counts_hosts_df[medal_counts_hosts_df['Year'] == year]
    
    # 2. 构建节点
    countries = year_data['NOC'].unique()
    for country in countries:
        G.add_node(country)

    # 3. 构建边权重（基于金牌差）
    for i, country1 in year_data.iterrows():
        for j, country2 in year_data.iterrows():
            if country1['NOC'] != country2['NOC']:  # 排除自己与自己的关系
                # 计算金牌差异作为边的权重
                gold_diff = abs(country1['Gold'] - country2['Gold'])
                if gold_diff > 0:  # 只有当金牌差大于0时才创建边
                    G.add_edge(country1['NOC'], country2['NOC'], weight=gold_diff)

    # 检查图的连通性，找出没有连接的节点
    isolated_nodes = [node for node, degree in G.degree() if degree == 0]
    print(f"Year {year}: 没有连接的节点: {isolated_nodes}")

    # 4. 计算中心性指标，并加上教练效应的调整
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')  # 加入权重计算介数中心性
    closeness_centrality = nx.closeness_centrality(G, distance='weight')  # 使用加权接近中心性

    # 5. 使用教练效应调整中心性
    for country in countries:
        # 获取教练效应 (C_t)
        coach_effect = year_data[year_data['NOC'] == country]['Coach_Effect'].values[0]  # 假设已计算出教练效应
        
        # 根据教练效应调整度中心性、介数中心性和接近中心性
        adjusted_degree_centrality = degree_centrality.get(country, 0) * np.log(1 + kappa_C * coach_effect * year_data[year_data['NOC'] == country]['predicted_NR'].values[0])
        adjusted_betweenness_centrality = betweenness_centrality.get(country, 0) * np.log(1 + mu_C * coach_effect * year_data[year_data['NOC'] == country]['predicted_NR'].values[0])
        adjusted_closeness_centrality = closeness_centrality.get(country, 0) * np.log(1 + nu_C * coach_effect * year_data[year_data['NOC'] == country]['predicted_NR'].values[0])

        country_data = {
            'Year': year,
            'Country': country,
            'Adjusted Degree Centrality': adjusted_degree_centrality,
            'Adjusted Betweenness Centrality': adjusted_betweenness_centrality,
            'Adjusted Closeness Centrality': adjusted_closeness_centrality
        }
        centrality_results.append(country_data)

# 6. 将所有结果保存到一个DataFrame
centrality_df = pd.DataFrame(centrality_results)

# 7. 导出结果到CSV文件
centrality_df.to_csv("centrality_analysis_with_coach_effect.csv", index=False)

# 输出中心性分析结果
print("每年每个国家的中心性分析已导出为 'centrality_analysis_with_coach_effect.csv'")
