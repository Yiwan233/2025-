# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题1：贡献者人数识别

代码名称: P1_V1.12_Complete_Workflow
版本: 1.12
日期: 2025-05-07
描述: 这是一个完整的脚本，整合了从数据加载、NoC提取、特征工程、
      特征可视化（可选）到基线模型和决策树模型评估的所有步骤。
      确保了 df_features 的正确生成和传递。
"""
# --- 基础库导入 ---
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from math import ceil # 用于基线模型

# --- 机器学习库导入 ---
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
# from sklearn.preprocessing import LabelEncoder # NoC_True 已是数字，暂不需要

# --- 配置与环境设置 ---
print("--- 脚本初始化与配置 (版本 1.12) ---")
# 忽略特定类型的警告
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Matplotlib 中文显示设置
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("INFO: Matplotlib 中文字体尝试设置为 'SimHei'.")
except Exception as e_font:
    print(f"警告: 设置 Matplotlib 中文字体 'SimHei' 失败: {e_font}. 图表中文可能无法显示。")

# 目录配置
DATA_DIR = './'
file_path_prob1 = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv') # 确保文件名正确
PLOTS_DIR = os.path.join(DATA_DIR, 'prob1_plots_v1.12_评估')
# feature_filename_prob1 = os.path.join(DATA_DIR, 'prob1_features_v1.12_wide.csv')

# --- 步骤 0：环境准备 ---
print(f"\n--- 步骤 0：环境准备 (版本 1.12) ---")
if not os.path.exists(PLOTS_DIR):
    try:
        os.makedirs(PLOTS_DIR)
        print(f"已创建绘图保存目录: {PLOTS_DIR}")
    except OSError as e_dir:
        print(f"错误：无法创建绘图目录 {PLOTS_DIR} : {e_dir}")
        PLOTS_DIR = DATA_DIR # 若失败则保存在当前目录

# --- 函数定义 (NoC 提取) ---
print(f"\n--- 函数定义 (版本 1.12) ---")
def extract_true_noc_v1_12(filename_str):
    """ 功能: 从文件名中提取真实的贡献者人数 (NoC) (版本 1.12) """
    filename_str = str(filename_str)
    match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', filename_str)
    if match:
        contributor_ids_str = match.group(1)
        ids_list = [id_val for id_val in contributor_ids_str.split('_') if id_val.isdigit()]
        num_contributors = len(ids_list)
        if num_contributors > 0:
            return int(num_contributors)
    return np.nan

# --- 步骤 1：数据加载与 NoC 提取 ---
print(f"\n--- 步骤 1：数据加载与 NoC 提取 (版本 1.12) ---")
df_prob1 = None
load_successful = False
try:
    print(f"尝试加载文件: '{file_path_prob1}' ...")
    df_prob1 = pd.read_csv(file_path_prob1, encoding='utf-8', sep=',', on_bad_lines='skip')
    print(f"成功加载文件。")
    load_successful = True
except FileNotFoundError:
    print(f"严重错误: 文件未找到 '{file_path_prob1}'。")
    load_successful = False
    exit() # 如果文件找不到，后续无法进行
except Exception as e_load:
    print(f"严重错误: 加载文件时发生意外错误: {e_load}")
    load_successful = False
    exit()

if df_prob1.empty:
    print("\n警告: 数据加载后 DataFrame 为空。脚本终止。")
    exit()
print(f"\n数据加载完成。 初始数据维度: {df_prob1.shape}")

# --- 步骤 1.1：提取真实贡献者人数 (NoC_True) ---
print("\n--- 步骤 1.1：提取真实贡献者人数 (NoC_True) (版本 1.12) ---")
try:
    if 'Sample File' not in df_prob1.columns:
         print(f"\n严重错误: 数据中缺少必需的 'Sample File' 列。")
         exit()

    print("正在创建 NoC 映射表...")
    unique_files = df_prob1['Sample File'].dropna().unique()
    noc_map = {filename: extract_true_noc_v1_12(filename) for filename in unique_files}
    df_prob1['NoC_True_Mapped'] = df_prob1['Sample File'].map(noc_map)

    if df_prob1['NoC_True_Mapped'].isnull().all():
        print("严重错误: 所有样本的 NoC 都未能成功提取。请检查 'extract_true_noc_v1_12' 函数和文件名格式。")
        print("DEBUG: noc_map 的前5项:", list(noc_map.items())[:5])
        print("DEBUG: 'NoC_True_Mapped' 值分布:", df_prob1['NoC_True_Mapped'].value_counts(dropna=False))
        exit()
    
    if df_prob1['NoC_True_Mapped'].isnull().any():
        num_failed_rows = df_prob1['NoC_True_Mapped'].isnull().sum()
        print(f"\n警告: {num_failed_rows} 行未能成功提取 NoC。正在删除这些行...")
        df_prob1.dropna(subset=['NoC_True_Mapped'], inplace=True)
        if df_prob1.empty:
             print("严重错误: 删除 NoC 提取失败的行后，DataFrame 为空。")
             exit()

    df_prob1['NoC_True'] = df_prob1['NoC_True_Mapped'].astype(int)
    df_prob1.drop(columns=['NoC_True_Mapped'], inplace=True)
    print("\n已成功处理 'NoC_True' 列。")
    print(f"包含有效 NoC_True 的数据维度: {df_prob1.shape}")

    noc_distribution = df_prob1['NoC_True'].value_counts().sort_index()
    print("\n最终检查: 清理后的 'NoC_True' 分布:")
    print(noc_distribution)
    if noc_distribution.empty or noc_distribution.index.max() <= 1:
         print(f"\n严重警告: 有效的 NoC 最大值小于等于 1。NoC 提取逻辑可能仍存在问题。")
         # exit() # 根据情况决定是否退出

except Exception as e_noc_extract:
     print(f"严重错误: 在 NoC 提取阶段发生错误: {e_noc_extract}")
     import traceback
     traceback.print_exc()
     exit()
print("--- 步骤 1 完成 ---")

# --- 步骤 2：特征工程 (宽格式数据) ---
if df_prob1.empty:
    print("\n错误: DataFrame df_prob1 为空，无法进行特征工程。")
    exit()
print(f"\n--- 步骤 2：特征工程 (版本 1.12) ---")
df_features = None # 初始化
try:
    def count_valid_alleles_in_row_v1_12(row):
        count = 0
        for i in range(1, 101):
            allele_col = f'Allele {i}'
            if allele_col in row.index and pd.notna(row[allele_col]):
                count += 1
        return count
    df_prob1['allele_count_per_marker'] = df_prob1.apply(count_valid_alleles_in_row_v1_12, axis=1)
    print("已计算每个位点的等位基因数。")

    print("正在聚合计算每个样本的特征...")
    grouped_by_sample_wide = df_prob1.groupby('Sample File')
    mac_per_sample = grouped_by_sample_wide['allele_count_per_marker'].max()
    if mac_per_sample.empty:
        print("错误: mac_per_sample 为空，无法继续。检查groupby操作。")
        exit()

    total_alleles_per_sample = grouped_by_sample_wide['allele_count_per_marker'].sum()
    avg_alleles_per_marker = grouped_by_sample_wide['allele_count_per_marker'].mean()
    markers_gt2 = grouped_by_sample_wide['allele_count_per_marker'].apply(lambda x: (x > 2).sum())
    markers_gt3 = grouped_by_sample_wide['allele_count_per_marker'].apply(lambda x: (x > 3).sum())
    markers_gt4 = grouped_by_sample_wide['allele_count_per_marker'].apply(lambda x: (x > 4).sum())

    def get_sample_height_stats_v1_12(group):
        heights = []
        for i in range(1, 101):
            height_col = f'Height {i}'
            if height_col in group.columns:
                numeric_heights = pd.to_numeric(group[height_col], errors='coerce')
                heights.extend(numeric_heights.dropna().tolist())
        if heights:
            return pd.Series({'avg_peak_height': np.mean(heights), 'std_peak_height': np.std(heights)})
        else:
            return pd.Series({'avg_peak_height': 0, 'std_peak_height': 0}) # 或者 np.nan
    height_stats = grouped_by_sample_wide.apply(get_sample_height_stats_v1_12)
    print("已聚合计算特征。")

    print("正在合并特征到最终的特征数据框 (df_features)...")
    unique_noc_map = df_prob1[['Sample File', 'NoC_True']].drop_duplicates().set_index('Sample File')['NoC_True']
    df_features = pd.DataFrame(index=mac_per_sample.index)
    df_features['NoC_True'] = df_features.index.map(unique_noc_map)

    if df_features['NoC_True'].isnull().any():
        print(f"警告: 特征数据框中发现 {df_features['NoC_True'].isnull().sum()} 个 NaN NoC_True。正在删除...")
        df_features.dropna(subset=['NoC_True'], inplace=True)
    if df_features.empty:
        print("严重错误: 特征数据框在处理 NoC 后为空。")
        exit()
    df_features['NoC_True'] = df_features['NoC_True'].astype(int)

    df_features['max_allele_per_sample'] = mac_per_sample
    df_features['total_alleles_per_sample'] = total_alleles_per_sample
    df_features['avg_alleles_per_marker'] = avg_alleles_per_marker
    df_features['markers_gt2_alleles'] = markers_gt2
    df_features['markers_gt3_alleles'] = markers_gt3
    df_features['markers_gt4_alleles'] = markers_gt4
    df_features['avg_peak_height'] = height_stats['avg_peak_height']
    df_features['std_peak_height'] = height_stats['std_peak_height']
    
    # Re-align features to df_features' index (in case some samples were dropped from NoC map)
    # This ensures all series added have the same index as df_features
    for col in ['max_allele_per_sample', 'total_alleles_per_sample', 'avg_alleles_per_marker', 
                'markers_gt2_alleles', 'markers_gt3_alleles', 'markers_gt4_alleles', 
                'avg_peak_height', 'std_peak_height']:
        if col in df_features.columns and col not in ['NoC_True']: # NoC_True is already set
             df_features[col] = df_features.index.map(locals()[col])


    df_features.fillna(0, inplace=True)
    df_features.reset_index(inplace=True)

    print("\n--- 特征工程完成 ---")
    print(f"最终特征数据框 df_features 维度: {df_features.shape}")
    if df_features.empty:
        print("严重错误: df_features 为空！")
        exit()
    print("\n特征数据框 (df_features) 前 5 行:")
    print(df_features.head())
    # print("\n特征数据框 (df_features) 的统计摘要:")
    # with pd.option_context('display.float_format', '{:.2f}'.format):
    #     print(df_features.drop(columns=['Sample File'], errors='ignore').describe())
    print("--- 步骤 2 完成 ---")

except Exception as e_feat_eng:
    print(f"严重错误: 在特征工程 (步骤 2) 阶段发生错误: {e_feat_eng}")
    import traceback
    traceback.print_exc()
    exit()


# --- 步骤 3：特征分布可视化 (可选，可注释掉以加速) ---
# RUN_VISUALIZATION = True # 改为 False 则跳过
RUN_VISUALIZATION = False # 改为 True 则运行
if RUN_VISUALIZATION:
    print(f"\n--- 步骤 3：特征分布可视化 (版本 1.12) ---")
    if df_features.empty or 'NoC_True' not in df_features.columns:
        print("警告: df_features 为空或缺少 'NoC_True' 列，跳过可视化。")
    else:
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
                feature_chn = feature_name_map_chinese.get(feature_eng, feature_eng)
                plt.figure(figsize=(10, 7))
                sns.boxplot(x='NoC_True', y=feature_eng, data=df_features, palette="viridis")
                sns.stripplot(x='NoC_True', y=feature_eng, data=df_features, color=".3", alpha=0.6, jitter=0.2)
                plt.title(f'{feature_chn} vs. 真实贡献者人数', fontsize=16)
                plt.xlabel('真实贡献者人数 (NoC_True)', fontsize=13)
                plt.ylabel(feature_chn, fontsize=13)
                plt.grid(True, linestyle='--', alpha=0.6)
                plot_filename = os.path.join(PLOTS_DIR, f'特征分布_{feature_eng}_vs_NoC.png')
                plt.savefig(plot_filename, dpi=150)
                # plt.show() # 如果在非交互环境，这行可能导致脚本暂停，可注释掉
                plt.close()
                print(f"已保存图表: {plot_filename}")
            except Exception as e_plot_viz:
                print(f"错误：绘制特征 '{feature_eng}' 的图表时出错: {e_plot_viz}")
        print("\n--- 步骤 3 完成 ---")
else:
    print("\n--- 步骤 3：特征分布可视化已跳过 ---")


# --- 步骤 4: 基线模型评估 (Heuristic: ceil(MAC/2)) ---
print(f"\n--- 步骤 4: 基线模型评估 (版本 1.12) ---")
if 'df_features' not in locals() or df_features.empty:
     print("\n错误: 特征数据框 'df_features' 未定义或为空，无法进行基线模型评估。")
else:
    if 'max_allele_per_sample' not in df_features.columns or 'NoC_True' not in df_features.columns:
        print("错误: df_features 中缺少 'max_allele_per_sample' 或 'NoC_True' 列。")
    else:
        try:
            df_features['baseline_pred'] = df_features['max_allele_per_sample'].apply(lambda x: ceil(x / 2))
            y_true_baseline = df_features['NoC_True']
            y_pred_baseline = df_features['baseline_pred']
            baseline_accuracy = accuracy_score(y_true_baseline, y_pred_baseline)
            print(f"\n基线模型 (ceil(MAC/2)) 评估结果:")
            print(f"  整体准确率 (Accuracy): {baseline_accuracy:.4f}")
            unique_labels = sorted(y_true_baseline.unique())
            print("\n  分类报告 (Classification Report):")
            print(classification_report(y_true_baseline, y_pred_baseline, labels=unique_labels, zero_division=0))
            cm_baseline = confusion_matrix(y_true_baseline, y_pred_baseline, labels=unique_labels)
            print("\n  混淆矩阵 (Confusion Matrix):")
            print(cm_baseline)
            try:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=unique_labels)
                disp.plot(cmap=plt.cm.Blues)
                plt.title('基线模型 (ceil(MAC/2)) 混淆矩阵')
                plot_filename = os.path.join(PLOTS_DIR, 'confusion_matrix_baseline.png')
                plt.savefig(plot_filename, dpi=150); plt.close()
                print(f"基线模型混淆矩阵图已保存: {plot_filename}")
            except Exception as e_cm_baseline_plot: print(f"警告：绘制基线混淆矩阵时出错: {e_cm_baseline_plot}")
        except Exception as e_baseline_eval: print(f"错误：评估基线模型时发生错误: {e_baseline_eval}")
    print("--- 步骤 4 完成 ---")


# --- 步骤 5: 机器学习模型 (决策树) 训练与评估 ---
print(f"\n--- 步骤 5: 机器学习模型 (决策树) (版本 1.12) ---")
if 'df_features' not in locals() or df_features.empty:
     print("\n错误: 特征数据框 'df_features' 未定义或为空，无法进行决策树模型评估。")
else:
    try:
        selected_features = [
            'max_allele_per_sample', 'avg_alleles_per_marker',
            'markers_gt3_alleles', 'markers_gt4_alleles'
        ]
        print(f"使用的特征: {selected_features}")
        missing_features = [f_name for f_name in selected_features if f_name not in df_features.columns]
        if missing_features:
            print(f"错误: 以下选择的特征在 df_features 中不存在: {missing_features}。")
        else:
            X = df_features[selected_features]
            y = df_features['NoC_True']
            if X.empty or y.empty:
                print("错误: 特征矩阵 X 或目标向量 y 为空。")
            else:
                dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=4)
                print(f"模型: 决策树分类器 (max_depth={dt_classifier.max_depth})")
                cv_strategy = LeaveOneOut()
                print(f"评估策略: 留一法交叉验证 (LOOCV)")
                print("正在执行交叉验证...")
                y_pred_dt_cv = cross_val_predict(dt_classifier, X, y, cv=cv_strategy)
                dt_accuracy_cv = accuracy_score(y, y_pred_dt_cv)
                print(f"\n决策树模型 LOOCV 评估结果:")
                print(f"  整体准确率 (Accuracy): {dt_accuracy_cv:.4f}")
                unique_labels_dt = sorted(y.unique())
                print("\n  分类报告 (Classification Report):")
                print(classification_report(y, y_pred_dt_cv, labels=unique_labels_dt, zero_division=0))
                cm_dt_cv = confusion_matrix(y, y_pred_dt_cv, labels=unique_labels_dt)
                print("\n  混淆矩阵 (Confusion Matrix):")
                print(cm_dt_cv)
                try:
                    disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt_cv, display_labels=unique_labels_dt)
                    disp_dt.plot(cmap=plt.cm.Blues)
                    plt.title(f'决策树 (深度={dt_classifier.max_depth}) LOOCV 混淆矩阵')
                    plot_filename = os.path.join(PLOTS_DIR, f'confusion_matrix_dt_loocv_depth{dt_classifier.max_depth}.png')
                    plt.savefig(plot_filename, dpi=150); plt.close()
                    print(f"决策树混淆矩阵图已保存: {plot_filename}")
                except Exception as e_cm_dt_plot: print(f"警告：绘制决策树混淆矩阵时出错: {e_cm_dt_plot}")

                print("\n--- 步骤 5.1 (可选): 可视化决策树规则 (在全部数据上训练) ---")
                try:
                    dt_classifier_vis = DecisionTreeClassifier(random_state=42, max_depth=dt_classifier.max_depth)
                    dt_classifier_vis.fit(X, y)
                    plt.figure(figsize=(22, 14)) # 调整尺寸以适应更深的树
                    plot_tree(dt_classifier_vis, feature_names=selected_features,
                              class_names=[str(c) for c in unique_labels_dt], filled=True,
                              rounded=True, proportion=False, precision=2, fontsize=10)
                    plt.title(f"决策树可视化 (全部数据训练, max_depth={dt_classifier.max_depth})", fontsize=16)
                    plot_filename = os.path.join(PLOTS_DIR, f'decision_tree_visualization_depth{dt_classifier.max_depth}.png')
                    plt.savefig(plot_filename, dpi=200); plt.close()
                    print(f"决策树可视化图像已保存: {plot_filename}")
                except Exception as e_tree_plot_vis: print(f"错误：可视化决策树时出错: {e_tree_plot_vis}")
    except Exception as e_dt_main:
        print(f"错误: 执行决策树模型训练或评估主流程时发生错误: {e_dt_main}")
        import traceback
        traceback.print_exc()
    print("--- 步骤 5 完成 ---")

# --- 脚本结束判断 ---
if 'df_features' not in locals() or df_features.empty:
     print("\n脚本执行因 'df_features' 未成功生成而提前中止。")

print(f"\n脚本 {os.path.basename(__file__)} (版本 1.12) 执行完毕。")