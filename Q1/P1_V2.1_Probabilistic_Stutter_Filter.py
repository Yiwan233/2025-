# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题1：贡献者人数识别

代码名称: P1_V2.2_Integrated_Stutter_Filter
版本: 2.2
日期: 2025-05-10
描述: 整合了完整的数据加载步骤 (定义 load_successful 和 df_prob1)，
      并实现了阶段二的核心逻辑：峰识别与Stutter概率化评估。
      输出处理后的峰数据 df_processed_peaks，其中包含CTA。
      后续的特征工程和模型评估需要基于 df_processed_peaks 重写。
"""
# --- 基础库与机器学习库导入 ---
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from math import ceil, exp, sqrt

# --- 配置与环境设置 (版本 2.2) ---
print("--- 脚本初始化与配置 (版本 2.2) ---")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("INFO: Matplotlib 中文字体尝试设置为 'SimHei'.")
except Exception as e_font:
    print(f"警告: 设置 Matplotlib 中文字体 'SimHei' 失败: {e_font}.")

DATA_DIR = './'
file_path_prob1 = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
PLOTS_DIR = os.path.join(DATA_DIR, 'prob1_plots_v2.2_stutter_filter')

if not os.path.exists(PLOTS_DIR):
    try:
        os.makedirs(PLOTS_DIR)
        print(f"已创建绘图保存目录: {PLOTS_DIR}")
    except OSError as e_dir:
        PLOTS_DIR = DATA_DIR

# --- 全局参数定义 (版本 2.2) ---
print(f"\n--- 全局参数定义 (版本 2.2) ---")
SATURATION_THRESHOLD = 30000
SIZE_TOLERANCE_BP = 0.5 # 用于Stutter位置判断
STUTTER_CV_HS_N_MINUS_1 = 0.25 # n-1 Stutter峰高的假设变异系数
GLOBAL_AT = 50 # 备用全局AT (如果Dye信息不可用或不匹配)
DYE_AT_VALUES = {
    'B': 75, 'BLUE': 75, 'G': 101, 'GREEN': 101, 'Y': 60, 'YELLOW': 60,
    'R': 69, 'RED': 69, 'P': 56, 'PURPLE': 56, 'O': 50, 'ORANGE': 50
}
# --- 中文翻译映射表定义 (版本 x.x) ---
COLUMN_TRANSLATION_MAP = {
    'Sample File': '样本文件',
    'NoC_True': '真实贡献人数',
    # ... (所有其他列名映射) ...
    'baseline_pred': '基线模型预测NoC'
}
DESCRIBE_INDEX_TRANSLATION_MAP = {
    'count': '计数', 'mean': '均值', # ... (所有describe索引映射) ...
}
CLASSIFICATION_REPORT_METRICS_MAP = {
    'precision': '精确率', 'recall': '召回率', 'f1-score': 'F1分数', 'support': '样本数'
}
CLASSIFICATION_REPORT_AVG_MAP = {
    'accuracy': '准确率(整体)', 'macro avg': '宏平均', 'weighted avg': '加权平均'
}
# MARKER_PARAMS 字典 (核心！您需要根据之前整理的结果完整填充)
# 这个字典的准确性和完整性对Stutter过滤至关重要
MARKER_PARAMS = {
    # 示例 (您需要用您表格中的所有Marker及其参数替换和补充):
    'D8S1179': {'L_repeat': 4, 'dye': 'R', 'model_type_n-1':'Allele Average', 'n_minus_1_m': 0.00461, 'n_minus_1_c': 0.01873, 'sr_avg_n-1': 0.08246, 'n_minus_1_sr_max': 0.20},
    'D21S11':  {'L_repeat': 4, 'dye': 'Y', 'model_type_n-1':'Allele Average', 'n_minus_1_m': 0.00523, 'n_minus_1_c': -0.06858,'sr_avg_n-1': 0.08990, 'n_minus_1_sr_max': 0.20},
    'D7S820':  {'L_repeat': 4, 'dye': 'Y', 'model_type_n-1':'Allele Regression', 'n_minus_1_m': 0.01048, 'n_minus_1_c': -0.05172, 'n_minus_1_sr_max': 0.20},
    'CSF1PO':  {'L_repeat': 4, 'dye': 'G', 'model_type_n-1':'Allele Regression', 'n_minus_1_m': 0.01144, 'n_minus_1_c': -0.05766, 'n_minus_1_sr_max': 0.20},
    'D3S1358': {'L_repeat': 4, 'dye': 'B', 'model_type_n-1':'Allele Average', 'n_minus_1_m': 0.01046, 'n_minus_1_c': -0.07057, 'sr_avg_n-1': 0.09860, 'n_minus_1_sr_max': 0.20},
    'TH01':    {'L_repeat': 4, 'dye': 'Y', 'model_type_n-1':'LUS Regression', 'n_minus_1_m': 0.00185, 'n_minus_1_c': 0.00801,  'n_minus_1_sr_max': 0.15}, # LUS可能简化为Allele Reg.
    'D13S317': {'L_repeat': 4, 'dye': 'B', 'model_type_n-1':'Allele Average', 'n_minus_1_m': 0.01161, 'n_minus_1_c': -0.07044, 'sr_avg_n-1': 0.05638, 'n_minus_1_sr_max': 0.20},
    'D16S539': {'L_repeat': 4, 'dye': 'G', 'model_type_n-1':'Allele Regression', 'n_minus_1_m': 0.01180, 'n_minus_1_c': -0.05950, 'n_minus_1_sr_max': 0.20},
    'D2S1338': {'L_repeat': 4, 'dye': 'G', 'model_type_n-1':'Allele Average', 'n_minus_1_m': 0.00490, 'n_minus_1_c': -0.01190, 'sr_avg_n-1': 0.09165, 'n_minus_1_sr_max': 0.20},
    'D19S433': {'L_repeat': 4, 'dye': 'R', 'model_type_n-1':'Allele Average', 'n_minus_1_m': 0.00981, 'n_minus_1_c': -0.06040, 'sr_avg_n-1': 0.08044, 'n_minus_1_sr_max': 0.20},
    'vWA':     {'L_repeat': 4, 'dye': 'Y', 'model_type_n-1':'Allele Average', 'n_minus_1_m': 0.01098, 'n_minus_1_c': -0.09819, 'sr_avg_n-1': 0.08928, 'n_minus_1_sr_max': 0.20}, #注意vWA 14 issue
    'TPOX':    {'L_repeat': 4, 'dye': 'Y', 'model_type_n-1':'Allele Regression', 'n_minus_1_m': 0.00611, 'n_minus_1_c': -0.02772, 'n_minus_1_sr_max': 0.15},
    'D18S51':  {'L_repeat': 4, 'dye': 'G', 'model_type_n-1':'Allele Regression', 'n_minus_1_m': 0.00879, 'n_minus_1_c': -0.04708, 'n_minus_1_sr_max': 0.20},
    'AMEL':    {'L_repeat': 0, 'dye': 'B', 'model_type_n-1':'N/A', 'n_minus_1_m': 0, 'n_minus_1_c': 0, 'n_minus_1_sr_max': 0},
    'D5S818':  {'L_repeat': 4, 'dye': 'Y', 'model_type_n-1':'Allele Average', 'n_minus_1_m': 0.01106, 'n_minus_1_c': -0.05569, 'sr_avg_n-1': 0.07378, 'n_minus_1_sr_max': 0.20},
    'FGA':     {'L_repeat': 4, 'dye': 'P', 'model_type_n-1':'Allele Average', 'n_minus_1_m': 0.00659, 'n_minus_1_c': -0.06731, 'sr_avg_n-1': 0.08296, 'n_minus_1_sr_max': 0.20},
    'D1S1656': {'L_repeat': 4, 'dye': 'B', 'model_type_n-1':'LUS Regression', 'n_minus_1_m': 0.00604, 'n_minus_1_c': 0.00132,  'n_minus_1_sr_max': 0.22},
    'D2S441':  {'L_repeat': 4, 'dye': 'B', 'model_type_n-1':'Allele Average', 'n_minus_1_m': -0.00002,'n_minus_1_c': 0.05096,  'sr_avg_n-1': 0.05268, 'n_minus_1_sr_max': 0.15},
    'D10S1248':{'L_repeat': 4, 'dye': 'B', 'model_type_n-1':'Allele Regression', 'n_minus_1_m': 0.01068, 'n_minus_1_c': -0.06292, 'n_minus_1_sr_max': 0.20},
    'Penta E': {'L_repeat': 5, 'dye': 'B', 'model_type_n-1':'Allele Regression', 'n_minus_1_m': 0.00398, 'n_minus_1_c': -0.01607, 'n_minus_1_sr_max': 0.10},
    'Penta D': {'L_repeat': 5, 'dye': 'G', 'model_type_n-1':'Allele Average', 'n_minus_1_m': 0.00213, 'n_minus_1_c': -0.00468, 'sr_avg_n-1': 0.01990, 'n_minus_1_sr_max': 0.05},
    'D12S391': {'L_repeat': 4, 'dye': 'R', 'model_type_n-1':'Allele Regression', 'n_minus_1_m': 0.01063, 'n_minus_1_c': -0.10533, 'n_minus_1_sr_max': 0.25},
    'SE33':    {'L_repeat': 4, 'dye': 'R', 'model_type_n-1':'Allele Average', 'n_minus_1_m': 0.00304, 'n_minus_1_c': 0.04811,  'sr_avg_n-1': 0.12304, 'n_minus_1_sr_max': 0.20},
    'D22S1045':{'L_repeat': 3, 'dye': 'R', 'model_type_n-1':'LUS Regression', 'n_minus_1_m': 0.01528, 'n_minus_1_c': -0.13540, 'n_minus_1_sr_max': 0.25},
    'DYS391':  {'L_repeat': 4, 'dye': 'P', 'model_type_n-1':'N/A', 'n_minus_1_m': 0, 'n_minus_1_c': 0, 'n_minus_1_sr_max': 0},
    'DYS576':  {'L_repeat': 4, 'dye': 'P', 'model_type_n-1':'N/A', 'n_minus_1_m': 0, 'n_minus_1_c': 0, 'n_minus_1_sr_max': 0},
    'DYS570':  {'L_repeat': 4, 'dye': 'P', 'model_type_n-1':'N/A', 'n_minus_1_m': 0, 'n_minus_1_c': 0, 'n_minus_1_sr_max': 0},
}
# 请务必检查并补全所有附件1中出现的Marker

# --- 函数定义 ---
print(f"\n--- 函数定义 (版本 2.2) ---")
def extract_true_noc_v2_2(filename_str): # 版本号更新
    filename_str = str(filename_str)
    match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', filename_str)
    if match:
        ids_list = [id_val for id_val in match.group(1).split('_') if id_val.isdigit()]
        return int(len(ids_list)) if len(ids_list) > 0 else np.nan
    return np.nan

def calculate_peak_confidence_v2_2(locus_peaks_df_input, marker_name, marker_params_dict,
                                 dye_at_dict, global_at_val, sat_threshold,
                                 size_tolerance, cv_hs_stutter): # 版本号更新, cv_hs_stutter
    """
    对单个位点的峰进行AT筛选、饱和度处理，并评估n-1 Stutter可能性，计算真实等位基因置信度。
    """
    # 创建副本以避免修改原始传入的locus_peaks_df_input片段
    locus_peaks_df = locus_peaks_df_input.copy()

    if marker_name not in marker_params_dict:
        # print(f"警告: Marker '{marker_name}' 在参数字典中未找到，跳过Stutter处理，仅进行AT筛选。")
        # 获取AT值
        current_dye = locus_peaks_df['Dye'].iloc[0] if not locus_peaks_df.empty and 'Dye' in locus_peaks_df.columns else 'UNKNOWN'
        at_val = dye_at_dict.get(current_dye.upper(), global_at_val) # 统一转大写匹配

        processed_peaks = []
        for _, peak_row in locus_peaks_df.iterrows():
            height_original = peak_row['Height']
            height_adj = min(float(height_original), float(sat_threshold)) # 确保为浮点数
            if height_adj >= at_val:
                try:
                    allele_numeric = float(peak_row['Allele'])
                except ValueError:
                    allele_numeric = np.nan
                processed_peaks.append({
                    'Allele': peak_row['Allele'], 'Allele_Numeric': allele_numeric,
                    'Size': peak_row['Size'], 'Height': height_adj,
                    'Original_Height': height_original, 'CTA': 1.0, # 未经stutter评估，置信度为1
                    'Is_Stutter_Suspect': False, 'Stutter_Score':0.0
                })
        return pd.DataFrame(processed_peaks)

    params = marker_params_dict[marker_name]
    l_repeat = params.get('L_repeat', 0)
    current_dye = locus_peaks_df['Dye'].iloc[0] if not locus_peaks_df.empty and 'Dye' in locus_peaks_df.columns else params.get('dye', 'UNKNOWN')
    at_val = dye_at_dict.get(current_dye.upper(), global_at_val)

    candidate_peaks_list = []
    for _, peak_row in locus_peaks_df.iterrows():
        height_original = float(peak_row['Height'])
        height_adj = min(height_original, float(sat_threshold))
        if height_adj >= at_val:
            try: allele_numeric = float(peak_row['Allele'])
            except ValueError: allele_numeric = np.nan
            candidate_peaks_list.append({
                'Allele': peak_row['Allele'], 'Allele_Numeric': allele_numeric,
                'Size': float(peak_row['Size']), 'Height': height_adj,
                'Original_Height': height_original,
                'Max_Prob_Is_Stutter': 0.0, # 这个峰作为stutter被其他峰解释的最大概率
                'Is_Stutter_Suspect': False # 是否被任何其他峰高度怀疑为stutter
            })
    
    if not candidate_peaks_list:
        return pd.DataFrame(columns=['Allele', 'Allele_Numeric', 'Size', 'Height', 'Original_Height', 'CTA', 'Is_Stutter_Suspect', 'Stutter_Score'])

    peaks_df = pd.DataFrame(candidate_peaks_list).sort_values(by='Height', ascending=False).reset_index(drop=True)
    
    if l_repeat == 0 or params.get('model_type_n-1') == 'N/A' or pd.isna(params.get('n_minus_1_sr_max')): # 如果是AMEL或无stutter参数
        peaks_df['CTA'] = 1.0
        peaks_df['Stutter_Score'] = 0.0 # Stutter评分为0
        return peaks_df[['Allele', 'Allele_Numeric', 'Size', 'Height', 'Original_Height', 'CTA', 'Is_Stutter_Suspect', 'Stutter_Score']]


    # n-1 Stutter 评估
    sr_max_n_minus_1 = params.get('n_minus_1_sr_max', 0.3) # 使用位点特异性或全局

    for i, pc_row in peaks_df.iterrows(): # pc_row 是潜在的stutter峰 (candidate)
        current_max_score_pc_is_stutter = 0.0
        for j, pp_row in peaks_df.iterrows(): # pp_row 是潜在的亲代峰 (parent)
            if i == j or pc_row['Height'] >= pp_row['Height']:
                continue

            is_pos_match_n_minus_1 = (abs((pp_row['Size'] - pc_row['Size']) - l_repeat) <= size_tolerance)
            
            if is_pos_match_n_minus_1:
                sr_obs = pc_row['Height'] / pp_row['Height'] if pp_row['Height'] > 0 else np.inf

                if sr_obs > sr_max_n_minus_1:
                    score_stutter_for_this_parent = 0.0
                else:
                    e_sr = 0.0
                    if pd.notna(pp_row['Allele_Numeric']): # 亲代等位基因必须是数字
                        if params.get('model_type_n-1') == 'Allele Regression':
                            e_sr = params.get('n_minus_1_m',0) * pp_row['Allele_Numeric'] + params.get('n_minus_1_c',0)
                        elif params.get('model_type_n-1') == 'LUS Regression (fallback to Allele Reg.)': # 简化LUS回退
                            e_sr = params.get('n_minus_1_m',0) * pp_row['Allele_Numeric'] + params.get('n_minus_1_c',0)
                        elif params.get('model_type_n-1') == 'Allele Average':
                            e_sr = params.get('sr_avg_n-1', params.get('n_minus_1_c', 0))
                    
                    e_sr = max(0.001, e_sr) # 避免E[SR]为0或负，设一个极小的正数底限
                    e_hs = e_sr * pp_row['Height']
                    
                    if e_hs > 1e-6 : # 避免E[Hs]过小
                        sigma_hs = cv_hs_stutter * e_hs
                        if sigma_hs > 1e-6:
                            z_score = (pc_row['Height'] - e_hs) / sigma_hs
                            current_score = exp(-0.5 * (z_score**2))
                            # 额外惩罚 SR_obs 显著高于 E[SR] 的情况
                            if sr_obs > e_sr * 1.8 : current_score *= 0.5 # 比预期高80%
                            if sr_obs > e_sr * 2.5 : current_score *= 0.1 # 比预期高150%
                        else:
                            current_score = 1.0 if abs(pc_row['Height'] - e_hs) < 1e-6 else 0.0
                    elif pc_row['Height'] < 1e-6 :
                        current_score = 1.0
                    else:
                        current_score = 0.0
                    score_stutter_for_this_parent = current_score
                
                current_max_score_pc_is_stutter = max(current_max_score_pc_is_stutter, score_stutter_for_this_parent)
        
        peaks_df.loc[i, 'Max_Prob_Is_Stutter'] = current_max_score_pc_is_stutter

    peaks_df['Is_Stutter_Suspect'] = peaks_df['Max_Prob_Is_Stutter'] >= 0.5 # 示例：Stutter分数大于等于0.5则高度怀疑
    peaks_df['CTA'] = 1.0 - peaks_df['Max_Prob_Is_Stutter']
    peaks_df['CTA'] = peaks_df['CTA'].clip(lower=0.0, upper=1.0)
    peaks_df.rename(columns={'Max_Prob_Is_Stutter': 'Stutter_Score'}, inplace=True) # 重命名列

    return peaks_df[['Allele', 'Allele_Numeric', 'Size', 'Height', 'Original_Height', 'CTA', 'Is_Stutter_Suspect', 'Stutter_Score']]


# --- 步骤 1: 数据加载与NoC提取 ---
# (与V1.15相同的代码块，确保 df_prob1 被正确加载和处理，NoC_True已提取)
print(f"\n--- 步骤 1：数据加载与 NoC 提取 (版本 2.2) ---")
df_prob1 = None; load_successful = False
try:
    df_prob1 = pd.read_csv(file_path_prob1, encoding='utf-8', sep=',', on_bad_lines='skip')
    load_successful = True; print(f"成功加载文件: '{file_path_prob1}'")
except Exception as e: print(f"加载文件失败: {e}"); exit()
if df_prob1.empty: print("数据加载后为空"); exit()
if 'Sample File' not in df_prob1.columns: print("'Sample File' 列不存在"); exit()
unique_files = df_prob1['Sample File'].dropna().unique()
noc_map = {filename: extract_true_noc_v2_2(filename) for filename in unique_files}
df_prob1['NoC_True_Mapped'] = df_prob1['Sample File'].map(noc_map)
df_prob1.dropna(subset=['NoC_True_Mapped'], inplace=True)
if df_prob1.empty: print("处理NoC后数据为空"); exit()
df_prob1['NoC_True'] = df_prob1['NoC_True_Mapped'].astype(int)
df_prob1.drop(columns=['NoC_True_Mapped'], inplace=True)
print(f"NoC 提取完成。数据维度: {df_prob1.shape}")
# ... (打印NoC分布的代码可以保留)

# --- 步骤 2: 峰识别与Stutter概率化评估 ---
print(f"\n--- 步骤 2: 峰识别与Stutter概率化评估 (版本 2.2) ---")
processed_peak_data_all_samples = []
if not MARKER_PARAMS: print("错误: MARKER_PARAMS 为空。"); exit()
    
sample_files_processed_count = 0
for sample_file_name, group_data_per_sample in df_prob1.groupby('Sample File'):
    sample_files_processed_count += 1
    # print(f"正在处理样本: {sample_file_name} ({sample_files_processed_count}/{len(df_prob1['Sample File'].unique())})")
    for marker_name_actual, locus_data_from_groupby in group_data_per_sample.groupby('Marker'):
        # 将宽格式的单行locus_data_from_groupby转换为长格式列表供函数处理
        current_locus_peaks_list = []
        # locus_data_from_groupby 此时对于每个Marker应该只有一行
        if locus_data_from_groupby.empty: continue
        row_marker_data = locus_data_from_groupby.iloc[0]

        for i in range(1, 101):
            allele_val = row_marker_data.get(f'Allele {i}')
            size_val = row_marker_data.get(f'Size {i}')
            height_val = row_marker_data.get(f'Height {i}')
            if pd.notna(allele_val) and pd.notna(size_val) and pd.notna(height_val):
                current_locus_peaks_list.append({
                    'Allele': allele_val, 'Size': size_val, 'Height': height_val,
                    'Dye': row_marker_data.get('Dye', 'UNKNOWN')
                })
        
        if not current_locus_peaks_list: continue
        locus_peaks_for_filter_df = pd.DataFrame(current_locus_peaks_list)
        
        processed_locus_df = calculate_peak_confidence_v2_2(
            locus_peaks_for_filter_df, marker_name_actual, MARKER_PARAMS,
            DYE_AT_VALUES, GLOBAL_AT, SATURATION_THRESHOLD,
            SIZE_TOLERANCE_BP, STUTTER_CV_HS_N_MINUS_1
        )
        if not processed_locus_df.empty:
            processed_locus_df['Sample File'] = sample_file_name
            processed_locus_df['Marker'] = marker_name_actual
            processed_peak_data_all_samples.append(processed_locus_df)

if not processed_peak_data_all_samples: print("错误: 处理所有样本后，没有有效的峰数据。"); exit()
df_processed_peaks = pd.concat(processed_peak_data_all_samples, ignore_index=True)
print(f"峰处理与Stutter评估完成。共处理 {sample_files_processed_count} 个样本文件。")
print("处理后的峰数据 (df_processed_peaks) 前5行示例 (包含CTA - 真实等位基因置信度):")
# 使用英文列名打印，因为后续特征工程依赖英文名
print(df_processed_peaks.head())


# --- 步骤 3: 特征工程 (版本 2.2 - 基于 df_processed_peaks) ---
print(f"\n--- 步骤 3: 特征工程 (版本 2.2) ---")
df_features_v2_2 = pd.DataFrame()
try:
    CONFIDENCE_THRESHOLD_FOR_TRUE_ALLELE = 0.5 # 真实等位基因的置信度阈值 (可调)
    print(f"用于筛选真实等位基因的置信度阈值 (CTA_threshold): {CONFIDENCE_THRESHOLD_FOR_TRUE_ALLELE}")
    
    df_processed_peaks['Is_Effective_Allele'] = df_processed_peaks['CTA'] >= CONFIDENCE_THRESHOLD_FOR_TRUE_ALLELE
    df_effective_alleles = df_processed_peaks[df_processed_peaks['Is_Effective_Allele']]
    
    if df_effective_alleles.empty:
        print("警告: 应用CTA阈值后，没有剩余有效等位基因。特征将主要为0。")
    
    # 确保后续groupby的索引是 'Sample File'
    all_sample_files_index_df = df_prob1[['Sample File', 'NoC_True']].drop_duplicates().set_index('Sample File')
    df_features_v2_2 = pd.DataFrame(index=all_sample_files_index_df.index)
    df_features_v2_2['NoC_True'] = all_sample_files_index_df['NoC_True'] # 继承正确的NoC_True

    # 计算 N_eff_alleles_per_locus
    if not df_effective_alleles.empty:
        n_eff_alleles_per_locus = df_effective_alleles.groupby(['Sample File', 'Marker'])['Allele'].count().rename('N_eff_alleles')
        grouped_by_sample_eff = n_eff_alleles_per_locus.groupby('Sample File')
        # 为df_features_v2_2添加新特征列，使用.map()或直接赋值（如果索引已对齐）
        df_features_v2_2['max_allele_per_sample'] = grouped_by_sample_eff.max()
        df_features_v2_2['total_alleles_per_sample'] = grouped_by_sample_eff.sum()
        df_features_v2_2['avg_alleles_per_marker'] = grouped_by_sample_eff.mean()
        df_features_v2_2['markers_gt2_alleles'] = grouped_by_sample_eff.apply(lambda x: (x > 2).sum())
        df_features_v2_2['markers_gt3_alleles'] = grouped_by_sample_eff.apply(lambda x: (x > 3).sum())
        df_features_v2_2['markers_gt4_alleles'] = grouped_by_sample_eff.apply(lambda x: (x > 4).sum())
        
        grouped_heights_eff = df_effective_alleles.groupby('Sample File')['Height']
        df_features_v2_2['avg_peak_height'] = grouped_heights_eff.mean()
        df_features_v2_2['std_peak_height'] = grouped_heights_eff.std()
        # PHR, Skew, Kurt等高级特征的计算逻辑可以基于 df_effective_alleles 在这里添加
    else: # 如果没有有效等位基因，则特征赋0
        for col in ['max_allele_per_sample', 'total_alleles_per_sample', 'avg_alleles_per_marker', 
                    'markers_gt2_alleles', 'markers_gt3_alleles', 'markers_gt4_alleles',
                    'avg_peak_height', 'std_peak_height']:
            df_features_v2_2[col] = 0
            
    df_features_v2_2.fillna(0, inplace=True)
    df_features_v2_2.reset_index(inplace=True)

    print("\n--- 特征工程 (基于概率化Stutter处理) 完成 ---")
    print(f"最终特征数据框 df_features_v2_2 维度: {df_features_v2_2.shape}")
    print_df_in_chinese(df_features_v2_2.head(), col_map=COLUMN_TRANSLATION_MAP, title="新特征数据框 (df_features_v2_2) 前5行")
    described_features_v2_2 = df_features_v2_2.drop(columns=['Sample File'], errors='ignore').describe()
    print_df_in_chinese(described_features_v2_2, col_map=COLUMN_TRANSLATION_MAP, index_item_map=DESCRIBE_INDEX_TRANSLATION_MAP, title="新特征数据框的统计摘要")

except Exception as e_feat_eng_v2_2:
    print(f"严重错误: 在特征工程 (步骤 3 - V2.2) 阶段发生错误: {e_feat_eng_v2_2}"); import traceback; traceback.print_exc(); exit()


# --- 步骤 4 & 5: 模型评估 (使用新的 df_features_v2_2) ---
# print(f"\n--- 步骤 4 & 5: 模型评估 (版本 2.2) ---")
# print("注意：后续模型评估将使用新的 df_features_v2_2 特征集。")
# (将 V1.15 中的步骤4和5代码复制到此处，确保输入是 df_features_v2_2)

print(f"\n脚本 {os.path.basename(__file__)} (版本 2.2) 核心Stutter过滤和特征工程框架完成。")
print("请完整填充 MARKER_PARAMS 字典, 并将之前版本中完整的【步骤1】以及【步骤4、5】代码整合。")
# --- 辅助函数：打印中文 DataFrame (版本 x.x) ---
