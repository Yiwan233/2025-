# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题1：贡献者人数识别

代码名称: P1_V1.13_Fix_Feature_Assignment
版本: 1.13
日期: 2025-05-07
描述: 修正了 V1.12 版本中特征工程步骤的一个错误。
      移除了不必要且导致 KeyError 的特征重新对齐循环。
      保留了正确的特征直接赋值方式。
      其他步骤（数据加载、NoC提取、可视化、模型评估）逻辑不变。
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

# --- 配置与环境设置 ---
print("--- 脚本初始化与配置 (版本 1.13) ---")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("INFO: Matplotlib 中文字体尝试设置为 'SimHei'.")
except Exception as e_font:
    print(f"警告: 设置 Matplotlib 中文字体 'SimHei' 失败: {e_font}. 图表中文可能无法显示。")

DATA_DIR = './'
file_path_prob1 = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
PLOTS_DIR = os.path.join(DATA_DIR, 'prob1_plots_v1.13_评估')
# feature_filename_prob1 = os.path.join(DATA_DIR, 'prob1_features_v1.13_wide.csv')

print(f"\n--- 步骤 0：环境准备 (版本 1.13) ---")
if not os.path.exists(PLOTS_DIR):
    try:
        os.makedirs(PLOTS_DIR)
        print(f"已创建绘图保存目录: {PLOTS_DIR}")
    except OSError as e_dir:
        print(f"错误：无法创建绘图目录 {PLOTS_DIR} : {e_dir}")
        PLOTS_DIR = DATA_DIR

print(f"\n--- 函数定义 (版本 1.13) ---")
def extract_true_noc_v1_13(filename_str): # Renamed to reflect script version
    filename_str = str(filename_str)
    match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', filename_str)
    if match:
        contributor_ids_str = match.group(1)
        ids_list = [id_val for id_val in contributor_ids_str.split('_') if id_val.isdigit()]
        num_contributors = len(ids_list)
        if num_contributors > 0:
            return int(num_contributors)
    return np.nan

print(f"\n--- 步骤 1：数据加载与 NoC 提取 (版本 1.13) ---")
df_prob1 = None
load_successful = False
try:
    print(f"尝试加载文件: '{file_path_prob1}' ...")
    df_prob1 = pd.read_csv(file_path_prob1, encoding='utf-8', sep=',', on_bad_lines='skip')
    print(f"成功加载文件。")
    load_successful = True
except FileNotFoundError:
    print(f"严重错误: 文件未找到 '{file_path_prob1}'。")
    exit()
except Exception as e_load:
    print(f"严重错误: 加载文件时发生意外错误: {e_load}")
    exit()

if df_prob1.empty:
    print("\n警告: 数据加载后 DataFrame 为空。脚本终止。")
    exit()
print(f"\n数据加载完成。 初始数据维度: {df_prob1.shape}")

print("\n--- 步骤 1.1：提取真实贡献者人数 (NoC_True) (版本 1.13) ---")
try:
    if 'Sample File' not in df_prob1.columns:
         print(f"\n严重错误: 数据中缺少必需的 'Sample File' 列。")
         exit()
    print("正在创建 NoC 映射表...")
    unique_files = df_prob1['Sample File'].dropna().unique()
    noc_map = {filename: extract_true_noc_v1_13(filename) for filename in unique_files}
    df_prob1['NoC_True_Mapped'] = df_prob1['Sample File'].map(noc_map)

    if df_prob1['NoC_True_Mapped'].isnull().all():
        print("严重错误: 所有样本的 NoC 都未能成功提取。请检查 'extract_true_noc_v1_13' 函数和文件名格式。")
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
    if noc_distribution.empty or (len(noc_distribution) > 0 and noc_distribution.index.max() <= 1):
         print(f"\n严重警告: 有效的 NoC 最大值小于等于 1 或分布为空。NoC 提取逻辑可能仍存在问题。")
except Exception as e_noc_extract:
     print(f"严重错误: 在 NoC 提取阶段发生错误: {e_noc_extract}")
     import traceback; traceback.print_exc(); exit()
print("--- 步骤 1 完成 ---")

if df_prob1.empty:
    print("\n错误: DataFrame df_prob1 在步骤1后为空，无法进行特征工程。")
    exit()
print(f"\n--- 步骤 2：特征工程 (版本 1.13) ---")
df_features = pd.DataFrame() # Initialize df_features
try:
    def count_valid_alleles_in_row_v1_13(row): # Renamed
        count = 0
        for i in range(1, 101):
            allele_col = f'Allele {i}'
            if allele_col in row.index and pd.notna(row[allele_col]): count += 1
        return count
    df_prob1['allele_count_per_marker'] = df_prob1.apply(count_valid_alleles_in_row_v1_13, axis=1)
    print("已计算每个位点的等位基因数。")

    print("正在聚合计算每个样本的特征...")
    grouped_by_sample_wide = df_prob1.groupby('Sample File')
    mac_per_sample = grouped_by_sample_wide['allele_count_per_marker'].max()
    if mac_per_sample.empty:
        print("错误: mac_per_sample (最大等位基因数) 计算结果为空。检查groupby或源数据。")
        exit()

    total_alleles_per_sample = grouped_by_sample_wide['allele_count_per_marker'].sum()
    avg_alleles_per_marker = grouped_by_sample_wide['allele_count_per_marker'].mean()
    markers_gt2 = grouped_by_sample_wide['allele_count_per_marker'].apply(lambda x: (x > 2).sum())
    markers_gt3 = grouped_by_sample_wide['allele_count_per_marker'].apply(lambda x: (x > 3).sum())
    markers_gt4 = grouped_by_sample_wide['allele_count_per_marker'].apply(lambda x: (x > 4).sum())

    def get_sample_height_stats_v1_13(group): # Renamed
        heights = []
        for i in range(1, 101):
            height_col = f'Height {i}'
            if height_col in group.columns:
                numeric_heights = pd.to_numeric(group[height_col], errors='coerce')
                heights.extend(numeric_heights.dropna().tolist())
        return pd.Series({'avg_peak_height': np.mean(heights) if heights else 0,
                          'std_peak_height': np.std(heights) if heights else 0})
    height_stats = grouped_by_sample_wide.apply(get_sample_height_stats_v1_13)
    print("已聚合计算特征。")

    print("正在合并特征到最终的特征数据框 (df_features)...")
    unique_noc_map = df_prob1[['Sample File', 'NoC_True']].drop_duplicates().set_index('Sample File')['NoC_True']
    df_features = pd.DataFrame(index=mac_per_sample.index) # Use index from a valid grouped series
    df_features['NoC_True'] = df_features.index.map(unique_noc_map)

    if df_features['NoC_True'].isnull().any():
        print(f"警告: 特征数据框中发现 {df_features['NoC_True'].isnull().sum()} 个 NaN NoC_True。正在删除...")
        df_features.dropna(subset=['NoC_True'], inplace=True)
    if df_features.empty:
        print("严重错误: 特征数据框在处理 NoC 后为空。")
        exit()
    df_features['NoC_True'] = df_features['NoC_True'].astype(int)

    # 直接赋值，Pandas会自动按索引对齐
    df_features['max_allele_per_sample'] = mac_per_sample
    df_features['total_alleles_per_sample'] = total_alleles_per_sample
    df_features['avg_alleles_per_marker'] = avg_alleles_per_marker
    df_features['markers_gt2_alleles'] = markers_gt2
    df_features['markers_gt3_alleles'] = markers_gt3
    df_features['markers_gt4_alleles'] = markers_gt4
    df_features['avg_peak_height'] = height_stats['avg_peak_height']
    df_features['std_peak_height'] = height_stats['std_peak_height']

    # 确保所有添加的特征都正确对齐到df_features的索引上
    # (如果mac_per_sample的索引是df_features的权威索引，且其他series也是基于相同groupby生成，通常是对齐的)
    # 但如果df_features因NoC_True的NaN而被删减行，则需要重新对齐其他特征列到df_features的当前索引
    for col_name, series_data in {
        'max_allele_per_sample': mac_per_sample,
        'total_alleles_per_sample': total_alleles_per_sample,
        'avg_alleles_per_marker': avg_alleles_per_marker,
        'markers_gt2_alleles': markers_gt2,
        'markers_gt3_alleles': markers_gt3,
        'markers_gt4_alleles': markers_gt4,
        'avg_peak_height': height_stats['avg_peak_height'],
        'std_peak_height': height_stats['std_peak_height']
    }.items():
        if col_name in df_features.columns: # Column should already exist from above assignments
            df_features[col_name] = df_features.index.map(series_data)


    df_features.fillna(0, inplace=True) # 填充可能因对齐产生的NaN或计算本身的NaN
    df_features.reset_index(inplace=True) # 'Sample File' 变为列

    print("\n--- 特征工程完成 ---")
    print(f"最终特征数据框 df_features 维度: {df_features.shape}")
    if df_features.empty: print("严重错误: df_features 为空！"); exit()
    print("\n特征数据框 (df_features) 前 5 行:"); print(df_features.head())
    print("--- 步骤 2 完成 ---")
except Exception as e_feat_eng_main:
    print(f"严重错误: 在特征工程 (步骤 2) 阶段发生错误: {e_feat_eng_main}")
    import traceback; traceback.print_exc(); exit()


# --- 步骤 3：特征分布可视化 (可选) ---
RUN_VISUALIZATION = False # 改为 True 则运行
if RUN_VISUALIZATION:
    print(f"\n--- 步骤 3：特征分布可视化 (版本 1.13) ---")
    # ... (V1.12中的可视化代码，确保使用df_features) ...
    if not df_features.empty:
        # ... (V1.12中的可视化代码，确保使用df_features) ...
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
        for feature_eng in features_to_visualize:
            if feature_eng not in df_features.columns: continue
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
                plt.savefig(plot_filename, dpi=150); plt.close()
                print(f"已保存图表: {plot_filename}")
            except Exception as e_plot_single: print(f"错误：绘制特征 '{feature_eng}' 图表时出错: {e_plot_single}")
        print("\n--- 步骤 3 完成 ---")
else: print("\n--- 步骤 3：特征分布可视化已跳过 ---")


# --- 步骤 4: 基线模型评估 ---
print(f"\n--- 步骤 4: 基线模型评估 (版本 1.13) ---")
# ... (V1.12中的基线模型评估代码，确保使用df_features) ...
if 'df_features' not in locals() or df_features.empty:
     print("\n错误: 'df_features' 未定义或为空，跳过基线评估。")
else:
    # ... (V1.12中的基线模型评估代码，确保使用df_features) ...
    if 'max_allele_per_sample' not in df_features.columns or 'NoC_True' not in df_features.columns:
        print("错误: df_features 中缺少评估基线模型所需列。")
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
                disp.plot(cmap=plt.cm.Blues); plt.title('基线模型 (ceil(MAC/2)) 混淆矩阵')
                plot_filename = os.path.join(PLOTS_DIR, 'confusion_matrix_baseline.png')
                plt.savefig(plot_filename, dpi=150); plt.close()
                print(f"基线模型混淆矩阵图已保存: {plot_filename}")
            except Exception as e_cm_base_plot: print(f"警告：绘制基线混淆矩阵时出错: {e_cm_base_plot}")
        except Exception as e_baseline_main: print(f"错误：评估基线模型时发生错误: {e_baseline_main}")
    print("--- 步骤 4 完成 ---")


# --- 步骤 5: 机器学习模型 (决策树) ---
print(f"\n--- 步骤 5: 机器学习模型 (决策树) (版本 1.13) ---")
# ... (V1.12中的决策树模型评估代码，确保使用df_features) ...
if 'df_features' not in locals() or df_features.empty:
     print("\n错误: 'df_features' 未定义或为空，跳过决策树评估。")
else:
    # ... (V1.12中的决策树模型评估代码，确保使用df_features) ...
    try:
        selected_features = ['max_allele_per_sample', 'avg_alleles_per_marker', 'markers_gt3_alleles', 'markers_gt4_alleles']
        print(f"使用的特征: {selected_features}")
        missing_features_dt = [f_name for f_name in selected_features if f_name not in df_features.columns]
        if missing_features_dt: print(f"错误: 决策树所需特征在 df_features 中不存在: {missing_features_dt}。")
        else:
            X = df_features[selected_features]
            y = df_features['NoC_True']
            if not X.empty and not y.empty:
                dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=4)
                print(f"模型: 决策树分类器 (max_depth={dt_classifier.max_depth})")
                cv_strategy = LeaveOneOut(); print(f"评估策略: 留一法交叉验证 (LOOCV)")
                print("正在执行交叉验证...")
                y_pred_dt_cv = cross_val_predict(dt_classifier, X, y, cv=cv_strategy)
                dt_accuracy_cv = accuracy_score(y, y_pred_dt_cv)
                print(f"\n决策树模型 LOOCV 评估结果:  整体准确率 (Accuracy): {dt_accuracy_cv:.4f}")
                unique_labels_dt = sorted(y.unique())
                print("\n  分类报告 (Classification Report):")
                print(classification_report(y, y_pred_dt_cv, labels=unique_labels_dt, zero_division=0))
                cm_dt_cv = confusion_matrix(y, y_pred_dt_cv, labels=unique_labels_dt)
                print("\n  混淆矩阵 (Confusion Matrix):"); print(cm_dt_cv)
                try:
                    disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt_cv, display_labels=unique_labels_dt)
                    disp_dt.plot(cmap=plt.cm.Blues); plt.title(f'决策树 (深度={dt_classifier.max_depth}) LOOCV 混淆矩阵')
                    plot_filename = os.path.join(PLOTS_DIR, f'confusion_matrix_dt_loocv_depth{dt_classifier.max_depth}.png')
                    plt.savefig(plot_filename, dpi=150); plt.close()
                    print(f"决策树混淆矩阵图已保存: {plot_filename}")
                except Exception as e_cm_dtree_plot: print(f"警告：绘制决策树混淆矩阵时出错: {e_cm_dtree_plot}")

                print("\n--- 步骤 5.1 (可选): 可视化决策树规则 ---")
                try:
                    dt_classifier_vis = DecisionTreeClassifier(random_state=42, max_depth=dt_classifier.max_depth).fit(X, y)
                    plt.figure(figsize=(22, 14))
                    plot_tree(dt_classifier_vis, feature_names=selected_features, class_names=[str(c) for c in unique_labels_dt], 
                              filled=True, rounded=True, proportion=False, precision=2, fontsize=10)
                    plt.title(f"决策树可视化 (全部数据训练, max_depth={dt_classifier.max_depth})", fontsize=16)
                    plot_filename = os.path.join(PLOTS_DIR, f'decision_tree_visualization_depth{dt_classifier.max_depth}.png')
                    plt.savefig(plot_filename, dpi=200); plt.close()
                    print(f"决策树可视化图像已保存: {plot_filename}")
                except Exception as e_tree_plot_vis_main: print(f"错误：可视化决策树时出错: {e_tree_plot_vis_main}")
            else: print("错误: 特征矩阵 X 或目标向量 y 为空，无法训练决策树。")
    except Exception as e_dt_main_block:
        print(f"错误: 执行决策树模型主流程时发生错误: {e_dt_main_block}")
        import traceback; traceback.print_exc()
    print("--- 步骤 5 完成 ---")

print(f"\n脚本 {os.path.basename(__file__)} (版本 1.13) 执行完毕。")