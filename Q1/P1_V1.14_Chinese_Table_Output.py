# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题1：贡献者人数识别

代码名称: P1_V1.15_Consolidated_ChineseTables
版本: 1.15
日期: 2025-05-09 (根据当前时间)
描述: 这是一个完整的脚本，整合了从数据加载、NoC提取、特征工程、
      可选可视化到基线模型和决策树模型评估的所有步骤。
      确保 df_features 正确生成，并实现关键表格输出的中文本地化。
      修正了之前版本中打印 noc_distribution 时可能出现的 KeyError。
"""
# --- 基础库与机器学习库导入 ---
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from math import ceil # 用于基线模型

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# --- 配置与环境设置 (版本 1.15) ---
print("--- 脚本初始化与配置 (版本 1.15) ---")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Matplotlib 中文显示设置
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows 系统常用的宋体
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 或者微软雅黑
    # 对于 macOS, 可以尝试: plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    # 对于 Linux, 可能需要指定具体字体路径或安装字体如: sudo apt-get install fonts-wqy-zenhei
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    print("INFO: Matplotlib 中文字体尝试设置为 'SimHei'.")
except Exception as e_font:
    print(f"警告: 设置 Matplotlib 中文字体 'SimHei' 失败: {e_font}. 图表中文可能无法正常显示。")
    print("警告: 请确保您的系统已安装相应的支持中文的字体。")

# 目录配置
DATA_DIR = './' # 假设数据文件在脚本所在目录
file_path_prob1 = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv') # 确保文件名正确
PLOTS_DIR = os.path.join(DATA_DIR, 'prob1_plots_v1.15_中文评估') # 模型相关的图保存目录
# feature_filename_prob1 = os.path.join(DATA_DIR, 'prob1_features_v1.15_wide.csv') # (可选)保存特征的文件名

# --- 步骤 0：环境准备 (版本 1.15) ---
print(f"\n--- 步骤 0：环境准备 (版本 1.15) ---")
if not os.path.exists(PLOTS_DIR):
    try:
        os.makedirs(PLOTS_DIR)
        print(f"已创建绘图保存目录: {PLOTS_DIR}")
    except OSError as e_dir:
        print(f"错误：无法创建绘图目录 {PLOTS_DIR} : {e_dir}. 图表将保存在当前目录。")
        PLOTS_DIR = DATA_DIR # 若失败则保存在当前目录

# --- 中文翻译映射表 (版本 1.15) ---
print(f"\n--- 中文翻译映射表定义 (版本 1.15) ---")
COLUMN_TRANSLATION_MAP = {
    'Sample File': '样本文件',
    'NoC_True': '真实贡献人数',
    'max_allele_per_sample': '样本内最大等位基因数',
    'total_alleles_per_sample': '样本内总等位基因数',
    'avg_alleles_per_marker': '每位点平均等位基因数',
    'markers_gt2_alleles': '>2等位基因位点数',
    'markers_gt3_alleles': '>3等位基因位点数',
    'markers_gt4_alleles': '>4等位基因位点数',
    'avg_peak_height': '平均峰高(RFU)',
    'std_peak_height': '峰高标准差(RFU)',
    'baseline_pred': '基线模型预测NoC'
}
DESCRIBE_INDEX_TRANSLATION_MAP = { # 用于翻译 describe() 输出的行索引
    'count': '计数', 'mean': '均值', 'std': '标准差', 'min': '最小值',
    '25%': '25%分位数', '50%': '中位数(50%)', '75%': '75%分位数', 'max': '最大值'
}
CLASSIFICATION_REPORT_METRICS_MAP = { # 用于翻译分类报告的列名（指标）
    'precision': '精确率', 'recall': '召回率', 'f1-score': 'F1分数', 'support': '样本数'
}
CLASSIFICATION_REPORT_AVG_MAP = { # 用于翻译分类报告的行索引（平均指标和类别）
    'accuracy': '准确率(整体)', 'macro avg': '宏平均', 'weighted avg': '加权平均'
    # 类别标签（如 '2', '3'）会在后面动态加入并翻译
}

# --- 辅助函数：打印中文 DataFrame (版本 1.15) ---
def print_df_in_chinese(df_to_print, col_map=None, index_item_map=None, index_name_map=None, title="DataFrame 内容", float_format='{:.4f}'):
    """打印 DataFrame，可选列名、索引项和索引名翻译，并格式化浮点数"""
    print(f"\n{title}:")
    df_display = df_to_print.copy()
    if col_map:
        df_display.columns = [col_map.get(str(col), str(col)) for col in df_display.columns] #确保col是字符串
    if index_item_map:
         df_display.index = [index_item_map.get(str(idx), str(idx)) for idx in df_display.index] #确保idx是字符串
    if index_name_map and df_display.index.name is not None: # 仅当索引有名字时翻译
        df_display.index.name = index_name_map.get(str(df_display.index.name), str(df_display.index.name))
    
    with pd.option_context('display.float_format', float_format.format, 'display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df_display)

# --- 函数定义 (NoC 提取 - 版本 1.15) ---
print(f"\n--- 函数定义 (版本 1.15) ---")
def extract_true_noc_v1_15(filename_str): # 版本号更新
    """ 功能: 从文件名中提取真实的贡献者人数 (NoC) (版本 1.15) """
    filename_str = str(filename_str)
    match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', filename_str) # 正确的正则表达式
    if match:
        contributor_ids_str = match.group(1)
        ids_list = [id_val for id_val in contributor_ids_str.split('_') if id_val.isdigit()]
        num_contributors = len(ids_list)
        return int(num_contributors) if num_contributors > 0 else np.nan
    return np.nan

# --- 步骤 1：数据加载与 NoC 提取 (版本 1.15) ---
print(f"\n--- 步骤 1：数据加载与 NoC 提取 (版本 1.15) ---")
df_prob1 = None
load_successful = False
try:
    print(f"尝试加载文件: '{file_path_prob1}' ...")
    df_prob1 = pd.read_csv(file_path_prob1, encoding='utf-8', sep=',', on_bad_lines='skip')
    print(f"成功加载文件。")
    load_successful = True
except FileNotFoundError:
    print(f"严重错误: 文件未找到 '{file_path_prob1}'。脚本终止。")
    exit()
except Exception as e_load:
    print(f"严重错误: 加载文件时发生意外错误: {e_load}。脚本终止。")
    exit()

if df_prob1.empty:
    print("\n警告: 数据加载后 DataFrame 为空。脚本终止。")
    exit()
print(f"\n数据加载完成。 初始数据维度: {df_prob1.shape}")

print("\n--- 步骤 1.1：提取真实贡献者人数 (NoC_True) (版本 1.15) ---")
try:
    if 'Sample File' not in df_prob1.columns:
         print(f"\n严重错误: 数据中缺少必需的 'Sample File' 列。脚本终止。")
         exit()
    print("正在创建 NoC 映射表...")
    unique_files = df_prob1['Sample File'].dropna().unique()
    noc_map = {filename: extract_true_noc_v1_15(filename) for filename in unique_files} # 使用 v1.15 函数
    df_prob1['NoC_True_Mapped'] = df_prob1['Sample File'].map(noc_map)

    if df_prob1['NoC_True_Mapped'].isnull().all():
        print("严重错误: 所有样本的 NoC 都未能成功提取。请检查 'extract_true_noc_v1_15' 函数和文件名格式。")
        exit()
    if df_prob1['NoC_True_Mapped'].isnull().any():
        num_failed_rows = df_prob1['NoC_True_Mapped'].isnull().sum()
        print(f"\n警告: {num_failed_rows} 行未能成功提取 NoC。正在删除这些行...")
        df_prob1.dropna(subset=['NoC_True_Mapped'], inplace=True)
        if df_prob1.empty:
             print("严重错误: 删除 NoC 提取失败的行后，DataFrame 为空。脚本终止。")
             exit()
    df_prob1['NoC_True'] = df_prob1['NoC_True_Mapped'].astype(int)
    df_prob1.drop(columns=['NoC_True_Mapped'], inplace=True)
    print("\n已成功处理 'NoC_True' 列。")
    print(f"包含有效 NoC_True 的数据维度: {df_prob1.shape}")

    noc_distribution = df_prob1['NoC_True'].value_counts().sort_index()
    print("\n最终检查: 清理后的 'NoC_True' 分布:")
    noc_dist_df_display = noc_distribution.reset_index()
    noc_dist_df_display.columns = ['NoC_True', 'count'] # 使用英文列名，让辅助函数翻译
    # 将 'NoC_True' 列设置为索引，它的名字将是 'NoC_True'
    df_for_printing_noc_dist = noc_dist_df_display.set_index('NoC_True')
    print_df_in_chinese(df_for_printing_noc_dist,
                        col_map={'count': '样本数'},
                        index_name_map=COLUMN_TRANSLATION_MAP, # 用于翻译索引名 'NoC_True'
                        title="'NoC_True' 分布")
    if noc_distribution.empty or (len(noc_distribution) > 0 and noc_distribution.index.max() <= 1):
         print(f"\n严重警告: 有效的 NoC 最大值小于等于 1 或分布为空。NoC 提取逻辑可能仍存在问题。")
except Exception as e_noc_extract:
     print(f"严重错误: 在 NoC 提取阶段发生错误: {e_noc_extract}"); import traceback; traceback.print_exc(); exit()
print("--- 步骤 1 完成 ---")

# --- 步骤 2：特征工程 (宽格式数据) (版本 1.15) ---
if df_prob1.empty: print("\n错误: df_prob1 在步骤1后为空，无法进行特征工程。"); exit()
print(f"\n--- 步骤 2：特征工程 (版本 1.15) ---")
df_features = pd.DataFrame() # 初始化
try:
    def count_valid_alleles_in_row_v1_15(row): # 版本号更新
        count = 0
        for i in range(1, 101):
            if f'Allele {i}' in row.index and pd.notna(row[f'Allele {i}']): count += 1
        return count
    df_prob1['allele_count_per_marker'] = df_prob1.apply(count_valid_alleles_in_row_v1_15, axis=1)
    print("已计算每个位点的等位基因数。")

    print("正在聚合计算每个样本的特征...")
    grouped_by_sample_wide = df_prob1.groupby('Sample File')
    mac_per_sample = grouped_by_sample_wide['allele_count_per_marker'].max()
    if mac_per_sample.empty: print("错误: mac_per_sample 计算结果为空。"); exit()

    total_alleles_per_sample = grouped_by_sample_wide['allele_count_per_marker'].sum()
    avg_alleles_per_marker = grouped_by_sample_wide['allele_count_per_marker'].mean()
    markers_gt2 = grouped_by_sample_wide['allele_count_per_marker'].apply(lambda x: (x > 2).sum())
    markers_gt3 = grouped_by_sample_wide['allele_count_per_marker'].apply(lambda x: (x > 3).sum())
    markers_gt4 = grouped_by_sample_wide['allele_count_per_marker'].apply(lambda x: (x > 4).sum())

    def get_sample_height_stats_v1_15(group): # 版本号更新
        heights = []
        for i in range(1, 101):
            if f'Height {i}' in group.columns:
                numeric_heights = pd.to_numeric(group[f'Height {i}'], errors='coerce')
                heights.extend(numeric_heights.dropna().tolist())
        return pd.Series({'avg_peak_height': np.mean(heights) if heights else 0,
                          'std_peak_height': np.std(heights) if heights else 0})
    height_stats = grouped_by_sample_wide.apply(get_sample_height_stats_v1_15)
    print("已聚合计算特征。")

    print("正在合并特征到最终的特征数据框 (df_features)...")
    # unique_noc_map 的索引是 Sample File，值是 NoC_True
    unique_noc_map = df_prob1[['Sample File', 'NoC_True']].drop_duplicates().set_index('Sample File')['NoC_True']
    
    # df_features 的索引是 mac_per_sample.index (即 Sample File)
    df_features = pd.DataFrame(index=mac_per_sample.index)
    df_features['NoC_True'] = df_features.index.map(unique_noc_map)

    if df_features['NoC_True'].isnull().any():
        print(f"警告: df_features 中 'NoC_True' 列存在NaN ({df_features['NoC_True'].isnull().sum()}个)。正在删除这些行...")
        df_features.dropna(subset=['NoC_True'], inplace=True)
    if df_features.empty: print("严重错误: df_features 在处理 NoC 后为空。"); exit()
    df_features['NoC_True'] = df_features['NoC_True'].astype(int)

    # 直接赋值，Pandas 会基于 df_features 当前的索引进行对齐
    feature_series_map = {
        'max_allele_per_sample': mac_per_sample,
        'total_alleles_per_sample': total_alleles_per_sample,
        'avg_alleles_per_marker': avg_alleles_per_marker,
        'markers_gt2_alleles': markers_gt2,
        'markers_gt3_alleles': markers_gt3,
        'markers_gt4_alleles': markers_gt4,
        'avg_peak_height': height_stats['avg_peak_height'],
        'std_peak_height': height_stats['std_peak_height']
    }
    for col_name, series_data in feature_series_map.items():
        df_features[col_name] = series_data # Pandas aligns on index

    df_features.fillna(0, inplace=True)
    df_features.reset_index(inplace=True) # 'Sample File' 变为列

    print("\n--- 特征工程完成 ---")
    print(f"最终特征数据框 df_features 维度: {df_features.shape}")
    if df_features.empty: print("严重错误: df_features 为空！"); exit()
    
    print_df_in_chinese(df_features.head(), col_map=COLUMN_TRANSLATION_MAP, title="特征数据框 (df_features) 前5行")
    
    described_features = df_features.drop(columns=['Sample File'], errors='ignore').describe()
    print_df_in_chinese(described_features, col_map=COLUMN_TRANSLATION_MAP, index_item_map=DESCRIBE_INDEX_TRANSLATION_MAP, title="特征数据框 (df_features) 的统计摘要") # 使用 index_item_map
    print("--- 步骤 2 完成 ---")

except Exception as e_feat_eng_main:
    print(f"严重错误: 在特征工程 (步骤 2) 阶段发生错误: {e_feat_eng_main}"); import traceback; traceback.print_exc(); exit()


# --- 步骤 3：特征分布可视化 (可选) ---
RUN_VISUALIZATION = False # 设置为 True 来运行可视化
if RUN_VISUALIZATION and not df_features.empty:
    print(f"\n--- 步骤 3：特征分布可视化 (版本 1.15) ---")
    # ... (与V1.14/V1.10相同的绘图代码，确保标题和标签是中文) ...
    feature_name_map_chinese_plots = COLUMN_TRANSLATION_MAP #复用之前的定义
    features_to_visualize = [f for f in feature_name_map_chinese_plots.keys() if f not in ['Sample File', 'NoC_True', 'baseline_pred']]
    print(f"开始绘制特征分布图，将保存到: {PLOTS_DIR}")
    for feature_eng in features_to_visualize:
        if feature_eng not in df_features.columns: continue
        try:
            feature_chn = feature_name_map_chinese_plots.get(feature_eng, feature_eng)
            plt.figure(figsize=(10, 7))
            sns.boxplot(x='NoC_True', y=feature_eng, data=df_features, palette="viridis")
            sns.stripplot(x='NoC_True', y=feature_eng, data=df_features, color=".3", alpha=0.6, jitter=0.2)
            plt.title(f'{feature_chn} vs. 真实贡献者人数', fontsize=16)
            plt.xlabel('真实贡献者人数 (NoC_True)', fontsize=13)
            plt.ylabel(feature_chn, fontsize=13)
            plt.grid(True, linestyle='--', alpha=0.6)
            plot_filename = os.path.join(PLOTS_DIR, f'特征分布_{feature_eng}_vs_NoC.png')
            plt.savefig(plot_filename, dpi=150); plt.close()
            print(f"已保存图表: {plot_filename}")
        except Exception as e_plot_single: print(f"错误：绘制特征 '{feature_eng}' 图表时出错: {e_plot_single}")
    print("\n--- 步骤 3 完成 ---")
else: print("\n--- 步骤 3：特征分布可视化已跳过 ---")


# --- 步骤 4: 基线模型评估 ---
print(f"\n--- 步骤 4: 基线模型评估 (版本 1.15) ---")
if df_features.empty: print("\n错误: df_features 为空，跳过基线评估。");
else:
    if 'max_allele_per_sample' not in df_features.columns or 'NoC_True' not in df_features.columns:
        print("错误: df_features 中缺少评估基线模型所需列。")
    else:
        try:
            df_features['baseline_pred'] = df_features['max_allele_per_sample'].apply(lambda x: ceil(x / 2))
            y_true_b = df_features['NoC_True']
            y_pred_b = df_features['baseline_pred']
            
            df_baseline_preds_display = pd.DataFrame({
                'NoC_True': y_true_b,
                'baseline_pred': y_pred_b
            }).head()
            print_df_in_chinese(df_baseline_preds_display, col_map=COLUMN_TRANSLATION_MAP, title="基线模型预测示例")

            baseline_acc = accuracy_score(y_true_b, y_pred_b)
            print(f"\n基线模型 (ceil(MAC/2)) 评估结果:")
            print(f"  整体准确率: {baseline_acc:.4f}")
            
            labels_b = sorted(y_true_b.unique())
            target_names_b = [f'贡献人数 {i}' for i in labels_b]
            
            report_dict_b = classification_report(y_true_b, y_pred_b, labels=labels_b, target_names=target_names_b, output_dict=True, zero_division=0)
            report_df_b = pd.DataFrame(report_dict_b).transpose()
            # 构建动态的 target_names_map
            dynamic_target_names_map_b = {name: name for name in target_names_b} # Classes themselves
            combined_index_map_b = {**dynamic_target_names_map_b, **CLASSIFICATION_REPORT_AVG_MAP}
            report_df_b.rename(index=combined_index_map_b, columns=CLASSIFICATION_REPORT_METRICS_MAP, inplace=True)
            print_df_in_chinese(report_df_b, title="基线模型 分类报告", float_format='{:.2f}')

            cm_b = confusion_matrix(y_true_b, y_pred_b, labels=labels_b)
            cm_df_b = pd.DataFrame(cm_b, index=[f"真实 {i}" for i in labels_b], columns=[f"预测 {i}" for i in labels_b])
            print_df_in_chinese(cm_df_b, title="基线模型 混淆矩阵", float_format='{:.0f}')
            
            try:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_b, display_labels=target_names_b)
                disp.plot(cmap=plt.cm.Blues); plt.title('基线模型 混淆矩阵')
                plot_filename = os.path.join(PLOTS_DIR, 'confusion_matrix_baseline.png')
                plt.savefig(plot_filename, dpi=150); plt.close()
                print(f"基线模型混淆矩阵图已保存: {plot_filename}")
            except Exception as e_cm_base_plot: print(f"警告：绘制基线混淆矩阵时出错: {e_cm_base_plot}")
        except Exception as e_baseline_main: print(f"错误：评估基线模型时发生错误: {e_baseline_main}"); import traceback; traceback.print_exc()
    print("--- 步骤 4 完成 ---")


# --- 步骤 5: 机器学习模型 (决策树) ---
print(f"\n--- 步骤 5: 机器学习模型 (决策树) (版本 1.15) ---")
if df_features.empty: print("\n错误: df_features 为空，跳过决策树评估。");
else:
    try:
        selected_features = ['max_allele_per_sample', 'avg_alleles_per_marker', 'markers_gt3_alleles', 'markers_gt4_alleles']
        print(f"使用的特征 (英文原名): {selected_features}")
        print(f"使用的特征 (中文名): {[COLUMN_TRANSLATION_MAP.get(f, f) for f in selected_features]}")

        missing_features_dt = [f for f in selected_features if f not in df_features.columns]
        if missing_features_dt: print(f"错误: 决策树所需特征在 df_features 中不存在: {missing_features_dt}。")
        else:
            X = df_features[selected_features]
            y = df_features['NoC_True']
            if not X.empty and not y.empty:
                dt_clf = DecisionTreeClassifier(random_state=42, max_depth=4)
                print(f"模型: 决策树分类器 (最大深度={dt_clf.max_depth})")
                cv_strategy = LeaveOneOut(); print(f"评估策略: 留一法交叉验证 (LOOCV)")
                print("正在执行交叉验证...")
                y_pred_dt_cv = cross_val_predict(dt_clf, X, y, cv=cv_strategy)
                dt_acc_cv = accuracy_score(y, y_pred_dt_cv)
                print(f"\n决策树模型 LOOCV 评估结果:  整体准确率: {dt_acc_cv:.4f}")
                
                labels_dt = sorted(y.unique())
                target_names_dt = [f'贡献人数 {i}' for i in labels_dt]
                
                report_dict_dt = classification_report(y, y_pred_dt_cv, labels=labels_dt, target_names=target_names_dt, output_dict=True, zero_division=0)
                report_df_dt = pd.DataFrame(report_dict_dt).transpose()
                dynamic_target_names_map_dt = {name: name for name in target_names_dt}
                combined_index_map_dt = {**dynamic_target_names_map_dt, **CLASSIFICATION_REPORT_AVG_MAP}
                report_df_dt.rename(index=combined_index_map_dt, columns=CLASSIFICATION_REPORT_METRICS_MAP, inplace=True)
                print_df_in_chinese(report_df_dt, title=f"决策树模型 (max_depth={dt_clf.max_depth}) LOOCV 分类报告", float_format='{:.2f}')

                cm_dt = confusion_matrix(y, y_pred_dt_cv, labels=labels_dt)
                cm_df_dt = pd.DataFrame(cm_dt, index=[f"真实 {i}" for i in labels_dt], columns=[f"预测 {i}" for i in labels_dt])
                print_df_in_chinese(cm_df_dt, title=f"决策树模型 (max_depth={dt_clf.max_depth}) LOOCV 混淆矩阵", float_format='{:.0f}')
                
                try:
                    disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=target_names_dt)
                    disp_dt.plot(cmap=plt.cm.Blues); plt.title(f'决策树 (深度={dt_clf.max_depth}) LOOCV 混淆矩阵')
                    plot_filename = os.path.join(PLOTS_DIR, f'confusion_matrix_dt_loocv_depth{dt_clf.max_depth}.png')
                    plt.savefig(plot_filename, dpi=150); plt.close()
                    print(f"决策树混淆矩阵图已保存: {plot_filename}")
                except Exception as e_cm_dt_plot: print(f"警告：绘制决策树混淆矩阵时出错: {e_cm_dt_plot}")

                print("\n--- 步骤 5.1 (可选): 可视化决策树规则 ---")
                try:
                    dt_clf_vis = DecisionTreeClassifier(random_state=42, max_depth=dt_clf.max_depth).fit(X, y)
                    plt.figure(figsize=(24, 16))
                    feature_names_chinese_dt_vis = [COLUMN_TRANSLATION_MAP.get(f,f) for f in selected_features]
                    class_names_chinese_dt_vis = [str(c) + "人" for c in labels_dt]
                    plot_tree(dt_clf_vis, feature_names=feature_names_chinese_dt_vis, class_names=class_names_chinese_dt_vis, 
                              filled=True, rounded=True, proportion=False, precision=2, fontsize=10)
                    plt.title(f"决策树可视化 (全部数据训练, 最大深度={dt_clf.max_depth})", fontsize=16)
                    plot_filename = os.path.join(PLOTS_DIR, f'decision_tree_visualization_depth{dt_clf.max_depth}.png')
                    plt.savefig(plot_filename, dpi=200); plt.close()
                    print(f"决策树可视化图像已保存: {plot_filename}")
                except Exception as e_tree_plot_vis_main: print(f"错误：可视化决策树时出错: {e_tree_plot_vis_main}")
            else: print("错误: 特征矩阵 X 或目标向量 y 为空，无法训练决策树。")
    except Exception as e_dt_main_block:
        print(f"错误: 执行决策树模型主流程时发生错误: {e_dt_main_block}"); import traceback; traceback.print_exc()
    print("--- 步骤 5 完成 ---")

print(f"\n脚本 {os.path.basename(__file__)} (版本 1.15) 执行完毕。")