# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题1：贡献者人数识别

代码名称: P1_V1.10_Chinese_Localization
版本: 1.10
日期: 2025-05-06
描述: 基于 V1.8/V1.9 的逻辑，进行中文本地化处理。
      包括添加中文注释、中文打印输出、中文图表标签，
      并为主要逻辑步骤添加明确的标识和版本号。
      增加了 Matplotlib 的中文字体显示配置。
"""
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# --- 配置部分 (版本 1.10) ---
# 忽略特定类型的警告，保持输出整洁
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Matplotlib 中文显示设置
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 尝试使用 SimHei 字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    print("INFO: Matplotlib 中文字体尝试设置为 'SimHei'.")
except Exception as e:
    print(f"警告: 设置 Matplotlib 中文字体 'SimHei' 失败: {e}")
    print("警告: 图表中的中文可能无法正常显示。请确保您的系统已安装中文字体（如 SimHei, Microsoft YaHei 等），并根据需要修改 plt.rcParams['font.sans-serif'] 的设置。")

# 数据和绘图目录配置
DATA_DIR = './'  # 假设数据文件在脚本所在目录
file_path_prob1 = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv') # 确保文件名正确
PLOTS_DIR = os.path.join(DATA_DIR, 'prob1_plots_v1.10_中文') # 保存绘图的目录
feature_filename_prob1 = os.path.join(DATA_DIR, 'prob1_features_v1.10_wide.csv') # 保存特征的文件名 (可选)

# --- 步骤 0：环境准备 (版本 1.10) ---
print(f"--- 步骤 0：环境准备 (版本 1.10) ---")
# 创建绘图目录 (如果不存在)
if not os.path.exists(PLOTS_DIR):
    try:
        os.makedirs(PLOTS_DIR)
        print(f"已创建绘图保存目录: {PLOTS_DIR}")
    except OSError as e:
        print(f"错误：无法创建绘图目录 {PLOTS_DIR} : {e}")
        PLOTS_DIR = DATA_DIR # 如果创建失败，则保存在当前目录

# --- 函数定义 (版本 1.10) ---
print(f"\n--- 函数定义 (版本 1.10) ---")
def extract_true_noc_v1_10(filename_str):
    """
    函数名称: extract_true_noc_v1_10
    版本: 1.10
    功能: 从文件名中提取真实的贡献者人数 (NoC)。
          使用修正后的正则表达式，专门匹配 '-贡献者ID(s)-混合比例-' 的模式。
    输入: filename_str (str): 文件名字符串。
    返回: int: 贡献者人数，如果无法匹配则返回 np.nan。
    """
    filename_str = str(filename_str)
    # 正则表达式查找: 连字符 + (数字和下划线组合) + 连字符 + (数字和分号组合) + 连字符
    match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', filename_str)
    if match:
        contributor_ids_str = match.group(1)
        # 按下划线分割，并只保留纯数字的部分
        ids_list = [id_val for id_val in contributor_ids_str.split('_') if id_val.isdigit()]
        num_contributors = len(ids_list)
        if num_contributors > 0:
            return int(num_contributors)
        else:
            # 正则匹配成功，但分割后未找到有效数字ID（理论上不太可能）
            return np.nan
    else:
        # 未找到匹配模式
        return np.nan

# --- 步骤 1：数据加载与 NoC 提取 (版本 1.10) ---
print(f"\n--- 步骤 1：数据加载与 NoC 提取 (版本 1.10) ---")
df_prob1 = None
load_successful = False
try:
    print(f"尝试加载文件: '{file_path_prob1}' (编码: utf-8, 分隔符: ',', 跳过错误行)")
    df_prob1 = pd.read_csv(
        file_path_prob1,
        encoding='utf-8',
        sep=',',
        on_bad_lines='skip' # 跳过字段数错误的行
    )
    print(f"成功加载文件。")
    load_successful = True
except FileNotFoundError:
    print(f"严重错误: 文件未找到 '{file_path_prob1}'。请检查路径和文件名。")
    load_successful = False
except Exception as e:
    print(f"严重错误: 加载文件时发生意外错误: {e}")
    load_successful = False

# --- 仅在加载成功后继续 ---
if load_successful and df_prob1 is not None:
    if df_prob1.empty:
        print("\n警告: 数据加载后 DataFrame 为空。可能是所有行都被跳过或文件本身为空。脚本终止。")
        exit()
    print(f"\n数据加载完成。 初始数据维度: {df_prob1.shape}")

    # --- 步骤 1.1：提取真实贡献者人数 (NoC_True) ---
    print("\n--- 步骤 1.1：提取真实贡献者人数 (NoC_True) (版本 1.10) ---")
    try:
        if 'Sample File' not in df_prob1.columns:
             print(f"\n严重错误: 数据中缺少必需的 'Sample File' 列。无法提取 NoC。")
             exit()

        print("正在创建 NoC 映射表...")
        unique_files = df_prob1['Sample File'].dropna().unique()
        noc_map = {filename: extract_true_noc_v1_10(filename) for filename in unique_files}

        # DEBUG 输出 (检查映射结果)
        print("\nDEBUG: noc_map 中的前 10 项:")
        map_items_list = list(noc_map.items())
        for i, item in enumerate(map_items_list[:10]):
             print(f"  {i}: ('{item[0]}', {item[1]})")
        failed_in_map = sum(1 for noc_val in noc_map.values() if pd.isna(noc_val))
        print(f"DEBUG: NoC 映射失败的文件数 (值为 NaN): {failed_in_map}")


        df_prob1['NoC_True_Mapped'] = df_prob1['Sample File'].map(noc_map)

        print("\nDEBUG: 'NoC_True_Mapped' 列的值分布 (处理 NaN 和类型转换前):")
        print(df_prob1['NoC_True_Mapped'].value_counts(dropna=False))

        # 处理提取失败的样本 (NaN 值)
        if df_prob1['NoC_True_Mapped'].isnull().any():
            num_failed_rows = df_prob1['NoC_True_Mapped'].isnull().sum()
            print(f"\n警告: {num_failed_rows} 行未能成功提取 NoC (值为 NaN)。")
            unique_failed_samples = df_prob1[df_prob1['NoC_True_Mapped'].isnull()]['Sample File'].unique()
            print(f"未能提取 NoC 的独特样本文件 (最多显示 5 个，共 {len(unique_failed_samples)} 个):")
            for sample_file in unique_failed_samples[:5]: print(f"  - {sample_file}")
            print("正在删除这些 NoC 提取失败的行...")
            df_prob1.dropna(subset=['NoC_True_Mapped'], inplace=True)
            if df_prob1.empty:
                 print("严重错误: 删除 NoC 提取失败的行后，DataFrame 为空。")
                 exit()
            print(f"删除 NaN 行后的数据维度: {df_prob1.shape}")


        # 最终处理 NoC_True 列
        if not df_prob1.empty:
             df_prob1['NoC_True'] = df_prob1['NoC_True_Mapped'].astype(int)
             df_prob1.drop(columns=['NoC_True_Mapped'], inplace=True)
             print("\n已成功处理 'NoC_True' 列。")
        else:
             print("\n严重错误: DataFrame 为空，无法继续处理 'NoC_True'。")
             exit()

        # 最终检查 NoC 分布
        noc_distribution = df_prob1['NoC_True'].value_counts().sort_index()
        print("\n最终检查: 清理后的 'NoC_True' 分布:")
        print(noc_distribution)
        if noc_distribution.empty:
             print("\n严重错误: 没有有效的 NoC 数据！")
             exit()
        elif noc_distribution.index.max() <= 1:
             print(f"\n严重警告: 提取到的最大 NoC 小于等于 1。提取逻辑可能仍有问题！")

    except Exception as e_noc:
         print(f"严重错误: 在 NoC 提取阶段发生错误: {e_noc}")
         import traceback
         traceback.print_exc()
         exit()
    print("--- 步骤 1：数据加载与 NoC 提取完成 ---")

    # --- 步骤 2：特征工程 (宽格式数据) (版本 1.10) ---
    if df_prob1.empty:
        print("\n错误: DataFrame 为空，无法进行特征工程。")
        exit()

    print(f"\n--- 步骤 2：特征工程 (宽格式数据) (版本 1.10) ---")
    try:
        # 函数: 计算一行 (一个位点) 中的有效等位基因数
        def count_valid_alleles_in_row_v1_10(row):
            count = 0
            for i in range(1, 101): # 假设最多有 Allele 100 列
                allele_col = f'Allele {i}'
                if allele_col in row.index and pd.notna(row[allele_col]):
                    # 可在此添加过滤条件，例如: if row[allele_col] != 'OL':
                    count += 1
            return count
        df_prob1['allele_count_per_marker'] = df_prob1.apply(count_valid_alleles_in_row_v1_10, axis=1)
        print("已计算每个位点的等位基因数 ('allele_count_per_marker')。")

        print("正在聚合计算每个样本的特征...")
        grouped_by_sample_wide = df_prob1.groupby('Sample File')

        # 特征 1: 样本内最大等位基因数 (MAC)
        mac_per_sample = grouped_by_sample_wide['allele_count_per_marker'].max()
        # 特征 2: 样本内总等位基因数 (跨所有位点)
        total_alleles_per_sample = grouped_by_sample_wide['allele_count_per_marker'].sum()
        # 特征 3: 样本内平均每位点等位基因数
        avg_alleles_per_marker = grouped_by_sample_wide['allele_count_per_marker'].mean()
        # 特征 4-6: 等位基因数大于 X 的位点数
        markers_gt2 = grouped_by_sample_wide['allele_count_per_marker'].apply(lambda x: (x > 2).sum())
        markers_gt3 = grouped_by_sample_wide['allele_count_per_marker'].apply(lambda x: (x > 3).sum())
        markers_gt4 = grouped_by_sample_wide['allele_count_per_marker'].apply(lambda x: (x > 4).sum())

        # 函数: 计算样本的峰高统计信息
        def get_sample_height_stats_v1_10(group):
            heights = []
            for i in range(1, 101): # 假设最多有 Height 100 列
                height_col = f'Height {i}'
                if height_col in group.columns:
                    numeric_heights = pd.to_numeric(group[height_col], errors='coerce')
                    heights.extend(numeric_heights.dropna().tolist())
            if heights:
                return pd.Series({'avg_peak_height': np.mean(heights), 'std_peak_height': np.std(heights)})
            else:
                return pd.Series({'avg_peak_height': 0, 'std_peak_height': 0})
        height_stats = grouped_by_sample_wide.apply(get_sample_height_stats_v1_10)
        print("已聚合计算特征。")

        print("正在合并特征到最终的特征数据框 (df_features)...")
        # 获取每个独特样本的 NoC_True
        unique_noc_map = df_prob1[['Sample File', 'NoC_True']].drop_duplicates().set_index('Sample File')['NoC_True']

        # 创建特征数据框
        df_features = pd.DataFrame(index=mac_per_sample.index) # 使用 group 的索引 (Sample File)
        df_features['NoC_True'] = df_features.index.map(unique_noc_map) # 映射 NoC

        # 处理映射可能引入的 NaN，并设置类型
        if df_features['NoC_True'].isnull().any():
            print(f"警告: 特征数据框中发现 {df_features['NoC_True'].isnull().sum()} 个 NaN NoC_True。正在删除...")
            df_features.dropna(subset=['NoC_True'], inplace=True)
        if df_features.empty:
            print("严重错误: 特征数据框在处理 NoC 后为空。")
            exit()
        df_features['NoC_True'] = df_features['NoC_True'].astype(int)

        # 添加其他特征
        df_features['max_allele_per_sample'] = mac_per_sample
        df_features['total_alleles_per_sample'] = total_alleles_per_sample
        df_features['avg_alleles_per_marker'] = avg_alleles_per_marker
        df_features['markers_gt2_alleles'] = markers_gt2
        df_features['markers_gt3_alleles'] = markers_gt3
        df_features['markers_gt4_alleles'] = markers_gt4
        df_features['avg_peak_height'] = height_stats['avg_peak_height']
        df_features['std_peak_height'] = height_stats['std_peak_height']

        # Re-align and Fillna
        df_features = df_features.loc[mac_per_sample.index] #确保索引一致性（如果前面有dropna）
        df_features.fillna(0, inplace=True) # 填充剩余NaN（如std=NaN）
        df_features.reset_index(inplace=True) # 将 'Sample File' 从索引变为列

        print("\n--- 特征工程完成 ---")
        print(f"最终特征数据框 df_features 维度: {df_features.shape}")
        if df_features.empty:
            print("严重错误: df_features 为空！")
            exit()

        print("\n特征数据框 (df_features) 前 5 行:")
        print(df_features.head())
        print("\n特征数据框 (df_features) 的统计摘要:")
        with pd.option_context('display.float_format', '{:.2f}'.format):
            print(df_features.drop(columns=['Sample File'], errors='ignore').describe())

        print("--- 步骤 2：特征工程完成 ---")

        # --- 步骤 3：特征分布可视化 (版本 1.10) ---
        print(f"\n--- 步骤 3：特征分布可视化 (版本 1.10) ---")
        if df_features.empty or 'NoC_True' not in df_features.columns:
            print("警告: df_features 为空或缺少 'NoC_True' 列，跳过可视化。")
        else:
            # 定义特征英文名到中文名的映射
            feature_name_map_chinese = {
                'max_allele_per_sample': '样本内最大等位基因数 (MAC)',
                'total_alleles_per_sample': '样本内总等位基因数',
                'avg_alleles_per_marker': '样本内平均每位点等位基因数',
                'markers_gt2_alleles': '等位基因数>2的位点数',
                'markers_gt3_alleles': '等位基因数>3的位点数',
                'markers_gt4_alleles': '等位基因数>4的位点数',
                'avg_peak_height': '样本内平均峰高 (RFU)',
                'std_peak_height': '样本内峰高标准差 (RFU)'
            }
            features_to_visualize = list(feature_name_map_chinese.keys())

            if df_features['NoC_True'].nunique() < 2:
                print(f"警告: 数据中仅有 {df_features['NoC_True'].nunique()} 种 NoC 值，可视化区分度可能有限。")

            print(f"开始绘制特征分布图，将保存到: {PLOTS_DIR}")
            for feature_eng in features_to_visualize:
                if feature_eng not in df_features.columns:
                    print(f"警告: 特征 '{feature_eng}' 不在 df_features 中，跳过绘图。")
                    continue
                try:
                    feature_chn = feature_name_map_chinese.get(feature_eng, feature_eng) # 获取中文名，失败则用英文名
                    plt.figure(figsize=(10, 7)) # 设置图像大小

                    # 绘制箱线图
                    sns.boxplot(x='NoC_True', y=feature_eng, data=df_features, palette="viridis")

                    # # 或者绘制小提琴图 (注释掉上面一行，取消注释下面一行)
                    # sns.violinplot(x='NoC_True', y=feature_eng, data=df_features, palette="viridis")
                    
                    # 添加抖动图(jitter)或散点图(stripplot)可以更清晰地看到数据点分布
                    sns.stripplot(x='NoC_True', y=feature_eng, data=df_features, color=".3", alpha=0.6, jitter=0.2)


                    plt.title(f'{feature_chn} vs. 真实贡献者人数', fontsize=16)
                    plt.xlabel('真实贡献者人数 (NoC_True)', fontsize=13)
                    plt.ylabel(feature_chn, fontsize=13)
                    plt.grid(True, linestyle='--', alpha=0.6) # 添加网格线

                    # 保存图像文件
                    plot_filename = os.path.join(PLOTS_DIR, f'特征分布_{feature_eng}_vs_NoC.png')
                    plt.savefig(plot_filename, dpi=150) # 保存较高分辨率的图像
                    print(f"已保存图表: {plot_filename}")
                    plt.close() # 关闭当前图像，避免在某些环境（如Jupyter外）中累积显示

                except Exception as e_plot:
                    print(f"错误：绘制特征 '{feature_eng}' 的图表时出错: {e_plot}")
            print("\n--- 步骤 3：特征分布可视化完成 ---")

        # --- 步骤 4 (可选): 保存特征数据框 ---
        print(f"\n--- 步骤 4 (可选): 保存特征数据框 (版本 1.10) ---")
        if feature_filename_prob1 and not df_features.empty:
            try:
                # 使用 utf-8-sig 编码确保 Excel 能正确识别 UTF-8 CSV
                df_features.to_csv(feature_filename_prob1, index=False, encoding='utf-8-sig')
                print(f"特征数据框已成功保存至: '{feature_filename_prob1}'")
            except Exception as e_save:
                print(f"错误: 保存特征数据框至 '{feature_filename_prob1}' 时失败: {e_save}")
        else:
            print("未指定保存文件名或特征数据框为空，跳过保存。")

    except Exception as e_feat_eng:
        print(f"严重错误: 在特征工程 (步骤 2) 阶段发生错误: {e_feat_eng}")
        import traceback
        traceback.print_exc()
        exit()
else:
    print("\n脚本终止，因为数据加载失败或未能成功处理 (步骤 1)。")
    # exit() # 脚本自然结束

# --- 脚本结束 ---
print(f"\n脚本 {os.path.basename(__file__)} (版本 1.10) 执行完毕。")