# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题1：贡献者人数识别

代码名称: P1_V2.2_Integrated_Stutter_Filter
版本: 2.2
日期: 2025-05-10
描述: 整合了完整的数据加载步骤，并实现了阶段二的核心逻辑：
      峰识别与Stutter概率化评估 (主要针对n-1反向Stutter)。
      输出处理后的峰数据 df_processed_peaks，其中包含CTA。
      后续的特征工程和模型评估将基于 df_processed_peaks。
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
SATURATION_THRESHOLD = 30000.0  # RFU, 来自文献
SIZE_TOLERANCE_BP = 0.5     # 片段大小比较的容忍误差 (bp)
STUTTER_CV_HS_GLOBAL = 0.25 # 全局Stutter峰高的假设变异系数 (n-1反向Stutter)
# 可以为不同stutter类型定义不同的CV_HS，例如：
# STUTTER_CV_HS_N_MINUS_1 = 0.25
# STUTTER_CV_HS_N_PLUS_1 = 0.30 (正向stutter通常变异更大)
# --- 中文翻译映射表定义 (版本 x.x) ---
print(f"\n--- 中文翻译映射表定义 (版本 x.x) ---") # 版本号更新为您当前脚本的版本
COLUMN_TRANSLATION_MAP = {
    'Sample File': '样本文件', 'Marker': '标记', 'Dye': '染料',
    'Allele': '等位基因', 'Allele_Numeric': '数值型等位基因',
    'Size': '片段大小(bp)', 'Height': '峰高(RFU)', 'Original_Height': '原始峰高(RFU)',
    'CTA': '真实等位基因置信度', 'Is_Stutter_Suspect': '高度可疑Stutter',
    'Stutter_Score_as_N_minus_1': '作为n-1 Stutter的得分', # 或您Stutter得分列的实际名称
    'NoC_True': '真实贡献人数',
    'max_allele_per_sample': '样本内最大有效等位基因数',
    'total_alleles_per_sample': '样本内总有效等位基因数',
    'avg_alleles_per_marker': '每标记平均有效等位基因数',
    'markers_gt2_alleles': '有效等位基因数>2的标记数',
    'markers_gt3_alleles': '有效等位基因数>3的标记数',
    'markers_gt4_alleles': '有效等位基因数>4的标记数',
    'avg_peak_height': '有效等位基因平均峰高',
    'std_peak_height': '有效等位基因峰高标准差',
    'baseline_pred': '基线模型预测NoC'
    # 如果还有其他列需要翻译，请在此处添加
}
DESCRIBE_INDEX_TRANSLATION_MAP = {
    'count': '计数', 'mean': '均值', 'std': '标准差', 'min': '最小值',
    '25%': '25%分位数', '50%': '中位数(50%)', '75%': '75%分位数', 'max': '最大值'
}
CLASSIFICATION_REPORT_METRICS_MAP = {
    'precision': '精确率', 'recall': '召回率', 'f1-score': 'F1分数', 'support': '样本数'
}
CLASSIFICATION_REPORT_AVG_MAP = {
    'accuracy': '准确率(整体)', 'macro avg': '宏平均', 'weighted avg': '加权平均'
}
GLOBAL_AT = 50 # 备用全局AT (如果Dye信息不可用或不匹配)
DYE_AT_VALUES = {
    'B': 75, 'BLUE': 75,
    'G': 101, 'GREEN': 101,
    'Y': 60, 'YELLOW': 60,
    'R': 69, 'RED': 69,
    'P': 56, 'PURPLE': 56,
    'O': 50, 'ORANGE': 50,
    'UNKNOWN': GLOBAL_AT # 为未知染料设定一个默认AT
}

# MARKER_PARAMS 字典 (核心！需要您根据整理的结果完整填充)
# 结构: 'MarkerName': {'L_repeat': int, 'Dye': char,
#                     'n_minus_1_model_type': 'Allele Regression'/'Allele Average'/'LUS Regression (fallback)',
#                     'n_minus_1_m': float, 'n_minus_1_c': float (或 sr_avg),
#                     'n_minus_1_sr_max': float,
#                     (可选) 'n_plus_1_model_type': ..., 'n_plus_1_m': ..., ... }
MARKER_PARAMS = {
    # 您之前整理的参数表应在此处完整填充
    # 例如 (仅为示例，请务必用您从文献整理的精确值替换):
    'D8S1179': {'L_repeat': 4, 'Dye': 'R', 'n_minus_1_model_type':'Allele Average', 'n_minus_1_m': 0.0, 'n_minus_1_c': 0.08246, 'n_minus_1_sr_max': 0.20},
    'D21S11':  {'L_repeat': 4, 'Dye': 'Y', 'n_minus_1_model_type':'Allele Average', 'n_minus_1_m': 0.0, 'n_minus_1_c': 0.08990, 'n_minus_1_sr_max': 0.20},
    'D7S820':  {'L_repeat': 4, 'Dye': 'Y', 'n_minus_1_model_type':'Allele Regression', 'n_minus_1_m': 0.01048, 'n_minus_1_c': -0.05172, 'n_minus_1_sr_max': 0.20},
    'CSF1PO':  {'L_repeat': 4, 'Dye': 'G', 'n_minus_1_model_type':'Allele Regression', 'n_minus_1_m': 0.01144, 'n_minus_1_c': -0.05766, 'n_minus_1_sr_max': 0.20},
    'D3S1358': {'L_repeat': 4, 'Dye': 'B', 'n_minus_1_model_type':'Allele Average', 'n_minus_1_m': 0.0, 'n_minus_1_c': 0.09860, 'n_minus_1_sr_max': 0.20},
    'TH01':    {'L_repeat': 4, 'Dye': 'Y', 'n_minus_1_model_type':'LUS Regression (fallback)', 'n_minus_1_m': 0.00185, 'n_minus_1_c': 0.00801,  'n_minus_1_sr_max': 0.15},
    'D13S317': {'L_repeat': 4, 'Dye': 'B', 'n_minus_1_model_type':'Allele Average', 'n_minus_1_m': 0.0, 'n_minus_1_c': 0.05638, 'n_minus_1_sr_max': 0.20},
    'D16S539': {'L_repeat': 4, 'Dye': 'G', 'n_minus_1_model_type':'Allele Regression', 'n_minus_1_m': 0.01180, 'n_minus_1_c': -0.05950, 'n_minus_1_sr_max': 0.20},
    'D2S1338': {'L_repeat': 4, 'Dye': 'G', 'n_minus_1_model_type':'Allele Average', 'n_minus_1_m': 0.0, 'n_minus_1_c': 0.09165, 'n_minus_1_sr_max': 0.20},
    'D19S433': {'L_repeat': 4, 'Dye': 'R', 'n_minus_1_model_type':'Allele Average', 'n_minus_1_m': 0.0, 'n_minus_1_c': 0.08044, 'n_minus_1_sr_max': 0.20},
    'vWA':     {'L_repeat': 4, 'Dye': 'Y', 'n_minus_1_model_type':'Allele Average', 'n_minus_1_m': 0.0, 'n_minus_1_c': 0.08928, 'n_minus_1_sr_max': 0.20},
    'TPOX':    {'L_repeat': 4, 'Dye': 'Y', 'n_minus_1_model_type':'Allele Regression', 'n_minus_1_m': 0.00611, 'n_minus_1_c': -0.02772, 'n_minus_1_sr_max': 0.15},
    'D18S51':  {'L_repeat': 4, 'Dye': 'G', 'n_minus_1_model_type':'Allele Regression', 'n_minus_1_m': 0.00879, 'n_minus_1_c': -0.04708, 'n_minus_1_sr_max': 0.20},
    'AMEL':    {'L_repeat': 0, 'Dye': 'B', 'n_minus_1_model_type':'N/A', 'n_minus_1_m': 0, 'n_minus_1_c': 0, 'n_minus_1_sr_max': 0},
    'D5S818':  {'L_repeat': 4, 'Dye': 'Y', 'n_minus_1_model_type':'Allele Average', 'n_minus_1_m': 0.0, 'n_minus_1_c': 0.07378, 'n_minus_1_sr_max': 0.20},
    'FGA':     {'L_repeat': 4, 'Dye': 'P', 'n_minus_1_model_type':'Allele Average', 'n_minus_1_m': 0.0, 'n_minus_1_c': 0.08296, 'n_minus_1_sr_max': 0.20},
    'D1S1656': {'L_repeat': 4, 'Dye': 'B', 'n_minus_1_model_type':'LUS Regression (fallback)', 'n_minus_1_m': 0.00604, 'n_minus_1_c': 0.00132,  'n_minus_1_sr_max': 0.22},
    'D2S441':  {'L_repeat': 4, 'Dye': 'B', 'n_minus_1_model_type':'Allele Average', 'n_minus_1_m': 0.0,'n_minus_1_c': 0.05268,  'n_minus_1_sr_max': 0.15},
    'D10S1248':{'L_repeat': 4, 'Dye': 'B', 'n_minus_1_model_type':'Allele Regression', 'n_minus_1_m': 0.01068, 'n_minus_1_c': -0.06292, 'n_minus_1_sr_max': 0.20},
    'Penta E': {'L_repeat': 5, 'Dye': 'B', 'n_minus_1_model_type':'Allele Regression', 'n_minus_1_m': 0.00398, 'n_minus_1_c': -0.01607, 'n_minus_1_sr_max': 0.10},
    'Penta D': {'L_repeat': 5, 'Dye': 'G', 'n_minus_1_model_type':'Allele Average', 'n_minus_1_m': 0.0, 'n_minus_1_c': 0.01990, 'n_minus_1_sr_max': 0.05},
    'D12S391': {'L_repeat': 4, 'Dye': 'R', 'n_minus_1_model_type':'Allele Regression', 'n_minus_1_m': 0.01063, 'n_minus_1_c': -0.10533, 'n_minus_1_sr_max': 0.25},
    'SE33':    {'L_repeat': 4, 'Dye': 'R', 'n_minus_1_model_type':'Allele Average', 'n_minus_1_m': 0.0, 'n_minus_1_c': 0.12304,  'n_minus_1_sr_max': 0.20},
    'D22S1045':{'L_repeat': 3, 'Dye': 'R', 'n_minus_1_model_type':'LUS Regression (fallback)', 'n_minus_1_m': 0.01528, 'n_minus_1_c': -0.13540, 'n_minus_1_sr_max': 0.25},
    'DYS391':  {'L_repeat': 4, 'Dye': 'P', 'n_minus_1_model_type':'N/A', 'n_minus_1_m': 0, 'n_minus_1_c': 0, 'n_minus_1_sr_max': 0},
    'DYS576':  {'L_repeat': 4, 'Dye': 'P', 'n_minus_1_model_type':'N/A', 'n_minus_1_m': 0, 'n_minus_1_c': 0, 'n_minus_1_sr_max': 0},
    'DYS570':  {'L_repeat': 4, 'Dye': 'P', 'n_minus_1_model_type':'N/A', 'n_minus_1_m': 0, 'n_minus_1_c': 0, 'n_minus_1_sr_max': 0},
}
# 请确保此字典覆盖您数据中的所有Marker，并使用您从文献中整理的最佳参数。

# --- 函数定义 ---
print(f"\n--- 函数定义 (版本 2.2) ---")
def extract_true_noc_v2_2(filename_str):
    # ... (与V2.2版本中相同的实现) ...
    filename_str = str(filename_str)
    match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', filename_str)
    if match:
        ids_list = [id_val for id_val in match.group(1).split('_') if id_val.isdigit()]
        return int(len(ids_list)) if len(ids_list) > 0 else np.nan
    return np.nan

def get_allele_numeric(allele_val_str):
    """尝试将等位基因标签转换为数值，处理非数字和小数。"""
    try:
        return float(allele_val_str)
    except ValueError:
        # 可以增加对特定非数字标签（如'X', 'Y' for AMEL, 或 'OL'）的处理逻辑
        if isinstance(allele_val_str, str) and allele_val_str.upper() == 'X': return -1.0 # 示例：用特殊数值代表X
        if isinstance(allele_val_str, str) and allele_val_str.upper() == 'Y': return -2.0 # 示例：用特殊数值代表Y
        return np.nan
# --- 辅助函数：打印中文 DataFrame (版本 x.x) ---
def print_df_in_chinese(df_to_print, col_map=None, index_item_map=None, index_name_map=None, title="DataFrame 内容", float_format='{:.4f}'):
    # ... (V1.15中该函数的完整定义) ...
    print(f"\n{title}:")
    df_display = df_to_print.copy()
    if col_map:
        df_display.columns = [col_map.get(str(col), str(col)) for col in df_display.columns]
    if index_item_map and df_display.index.name is not None and df_display.index.name in index_item_map:
        pass
    elif index_item_map:
        df_display.index = [index_item_map.get(str(idx), str(idx)) for idx in df_display.index]
    if index_name_map and df_display.index.name is not None:
        df_display.index.name = index_name_map.get(str(df_display.index.name), str(df_display.index.name))
    with pd.option_context('display.float_format', float_format.format, 'display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
        print(df_display)
def calculate_peak_confidence_v2_2(locus_peaks_df_input, marker_name, marker_params_dict,
                                 dye_at_dict, global_at_val, sat_threshold,
                                 size_tolerance, cv_hs_n_minus_1_global):
    """
    对单个位点的峰进行AT筛选、饱和度处理，并评估n-1 Stutter可能性，计算真实等位基因置信度。
    """
    locus_peaks_df = locus_peaks_df_input.copy() # 操作副本

    # 获取位点参数
    if marker_name not in marker_params_dict:
        params = {'L_repeat': 0, 'Dye': locus_peaks_df['Dye'].iloc[0] if not locus_peaks_df.empty and 'Dye' in locus_peaks_df.columns else 'UNKNOWN', 'n_minus_1_model_type': 'N/A'}
        # print(f"警告: Marker '{marker_name}' 在参数字典中未找到。")
    else:
        params = marker_params_dict[marker_name]

    l_repeat = params.get('L_repeat', 0)
    current_dye = locus_peaks_df['Dye'].iloc[0] if not locus_peaks_df.empty and 'Dye' in locus_peaks_df.columns else params.get('Dye', 'UNKNOWN')
    at_val = dye_at_dict.get(current_dye.upper(), global_at_val)

    # 1. AT筛选和饱和度处理
    candidate_peaks_list = []
    for _, peak_row in locus_peaks_df.iterrows():
        height_original = float(peak_row['Height']) # 确保是浮点数
        height_adj = min(height_original, sat_threshold)
        if height_adj >= at_val:
            allele_numeric = get_allele_numeric(peak_row['Allele'])
            candidate_peaks_list.append({
                'Allele': peak_row['Allele'], 'Allele_Numeric': allele_numeric,
                'Size': float(peak_row['Size']), 'Height': height_adj,
                'Original_Height': height_original,
                'Stutter_Score_as_N_minus_1': 0.0, # 作为n-1 Stutter的最大得分
                # 可以为其他stutter类型添加类似列: 'Stutter_Score_as_N_plus_1': 0.0
            })
    
    if not candidate_peaks_list:
        return pd.DataFrame(columns=['Allele', 'Allele_Numeric', 'Size', 'Height', 'Original_Height', 'CTA'])

    peaks_df = pd.DataFrame(candidate_peaks_list).sort_values(by='Height', ascending=False).reset_index().rename(columns={'index': 'original_peak_index'})
    
    # 如果是AMEL或无stutter模型，直接返回CTA=1
    if l_repeat == 0 or params.get('n_minus_1_model_type') == 'N/A' or pd.isna(params.get('n_minus_1_sr_max')):
        peaks_df['CTA'] = 1.0
        return peaks_df[['Allele', 'Allele_Numeric', 'Size', 'Height', 'Original_Height', 'CTA']]

    # 2. n-1 Stutter 评估
    sr_max_n_minus_1 = params.get('n_minus_1_sr_max', 0.3) # 使用位点特异性或全局

    for i, pc_row in peaks_df.iterrows(): # pc_row 是潜在的stutter峰 (candidate)
        max_score_pc_is_stutter_n_minus_1 = 0.0
        for j, pp_row in peaks_df.iterrows(): # pp_row 是潜在的亲代峰 (parent)
            if pc_row['original_peak_index'] == pp_row['original_peak_index'] or pc_row['Height'] >= pp_row['Height']:
                continue # 不能是自身，且stutter通常更矮

            # a. 位置确认 (n-1 Stutter)
            is_pos_match_n_minus_1 = (abs((pp_row['Size'] - pc_row['Size']) - l_repeat) <= size_tolerance)
            
            if is_pos_match_n_minus_1:
                sr_obs = pc_row['Height'] / pp_row['Height'] if pp_row['Height'] > 1e-6 else np.inf

                if sr_obs > sr_max_n_minus_1:
                    score_stutter_for_this_parent = 0.0
                else:
                    e_sr = 0.0
                    if pd.notna(pp_row['Allele_Numeric']):
                        model_type = params.get('n_minus_1_model_type')
                        if model_type == 'Allele Regression' or model_type == 'LUS Regression (fallback)':
                            e_sr = params.get('n_minus_1_m',0) * pp_row['Allele_Numeric'] + params.get('n_minus_1_c',0)
                        elif model_type == 'Allele Average':
                            e_sr = params.get('n_minus_1_c', 0) # c 即为 sr_avg
                    
                    e_sr = max(0.001, e_sr) # 避免E[SR]为0或负
                    e_hs = e_sr * pp_row['Height']
                    
                    current_score = 0.0
                    if e_hs > 1e-6 :
                        sigma_hs = cv_hs_n_minus_1_global * e_hs # 使用全局CV
                        if sigma_hs > 1e-6:
                            z_score = (pc_row['Height'] - e_hs) / sigma_hs
                            current_score = exp(-0.5 * (z_score**2))
                            # 额外惩罚 SR_obs 显著高于 E[SR] 的情况
                            if sr_obs > e_sr * 1.8 and e_sr > 1e-6 : current_score *= 0.5
                            if sr_obs > e_sr * 2.5 and e_sr > 1e-6 : current_score *= 0.1
                        else: # sigma_hs 极小，要求严格匹配
                            current_score = 1.0 if abs(pc_row['Height'] - e_hs) < 1e-3 * e_hs else 0.0 # 允许0.1%的误差
                    elif pc_row['Height'] < 1e-6 : # 如果E[Hs]为0，只有当Hc也为0时才完全匹配
                        current_score = 1.0
                    
                    score_stutter_for_this_parent = current_score
                
                max_score_pc_is_stutter_n_minus_1 = max(max_score_pc_is_stutter_n_minus_1, score_stutter_for_this_parent)
        
        peaks_df.loc[i, 'Stutter_Score_as_N_minus_1'] = max_score_pc_is_stutter_n_minus_1
    
    # (此处可以添加对其他Stutter类型如n+1的评估逻辑，并综合所有stutter score)
    # 简化：当前CTA仅基于n-1 Stutter评分
    peaks_df['CTA'] = 1.0 - peaks_df['Stutter_Score_as_N_minus_1']
    peaks_df['CTA'] = peaks_df['CTA'].clip(lower=0.0, upper=1.0)
    
    # 添加一个简单的 Is_Stutter_Suspect 标记
    peaks_df['Is_Stutter_Suspect'] = peaks_df['Stutter_Score_as_N_minus_1'] >= 0.5 # 示例阈值

    return peaks_df[['Allele', 'Allele_Numeric', 'Size', 'Height', 'Original_Height', 'CTA', 'Is_Stutter_Suspect', 'Stutter_Score_as_N_minus_1']]


# --- 步骤 1: 数据加载与NoC提取 ---
print(f"\n--- 步骤 1：数据加载与 NoC 提取 (版本 2.2) ---")
# ... (与 V1.15 相同的完整数据加载和 NoC_True 提取代码块，确保 load_successful 和 df_prob1 被正确设置)
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
# ... (打印NoC分布的代码)

# --- 步骤 2: 峰识别与Stutter概率化评估 ---
print(f"\n--- 步骤 2: 峰识别与Stutter概率化评估 (版本 2.2) ---")
processed_peak_data_all_samples = []
if not MARKER_PARAMS: print("错误: MARKER_PARAMS 为空。"); exit()
    
sample_files_processed_count = 0
unique_sample_files_total = df_prob1['Sample File'].nunique()

for sample_file_name, group_data_per_sample in df_prob1.groupby('Sample File'):
    sample_files_processed_count += 1
    print(f"正在处理样本: {sample_file_name} ({sample_files_processed_count}/{unique_sample_files_total})")
    
    sample_all_loci_processed_peaks = []
    for marker_name_actual, locus_data_from_groupby in group_data_per_sample.groupby('Marker'):
        current_locus_peaks_list = []
        if locus_data_from_groupby.empty: continue
        row_marker_data = locus_data_from_groupby.iloc[0]

        for i in range(1, 101): # 从 Allele 1 到 Allele 100
            allele_val = row_marker_data.get(f'Allele {i}')
            size_val = row_marker_data.get(f'Size {i}')
            height_val = row_marker_data.get(f'Height {i}')
            if pd.notna(allele_val) and pd.notna(size_val) and pd.notna(height_val):
                current_locus_peaks_list.append({
                    'Allele': allele_val, 'Size': size_val, 'Height': height_val,
                    'Dye': row_marker_data.get('Dye', 'UNKNOWN') # 确保Dye列存在于df_prob1
                })
        
        if not current_locus_peaks_list: continue
        locus_peaks_for_filter_df = pd.DataFrame(current_locus_peaks_list)
        
        processed_locus_df = calculate_peak_confidence_v2_2(
            locus_peaks_for_filter_df, marker_name_actual, MARKER_PARAMS,
            DYE_AT_VALUES, GLOBAL_AT, SATURATION_THRESHOLD,
            SIZE_TOLERANCE_BP, STUTTER_CV_HS_GLOBAL # 使用全局CV_HS
        )
        if not processed_locus_df.empty:
            processed_locus_df['Sample File'] = sample_file_name # 添加样本和位点标识
            processed_locus_df['Marker'] = marker_name_actual
            sample_all_loci_processed_peaks.append(processed_locus_df)
    
    if sample_all_loci_processed_peaks:
        processed_peak_data_all_samples.extend(sample_all_loci_processed_peaks)


if not processed_peak_data_all_samples: print("错误: 处理所有样本后，没有有效的峰数据。"); exit()
df_processed_peaks = pd.concat(processed_peak_data_all_samples, ignore_index=True)
print(f"峰处理与Stutter评估完成。共处理 {sample_files_processed_count} 个独立样本。")
print(f"处理后的总峰条目数 (df_processed_peaks): {len(df_processed_peaks)}")
print_df_in_chinese(df_processed_peaks.head(), title="处理后峰数据示例 (df_processed_peaks)")


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


print(f"\n脚本 {os.path.basename(__file__)} (版本 2.2) 执行完毕（核心Stutter过滤逻辑已构建）。")
print("请务必用您之前版本中完整的【步骤1数据加载与NoC提取】代码替换脚本中的简化部分。")
print("请务必完整填充 MARKER_PARAMS 字典，特别是L_repeat, Stutter模型类型, m, c/sr_avg, sr_max。")
print("后续【步骤3特征工程】和【步骤4、5模型评估】代码需要从之前版本整合，并确保使用本步骤产出的 df_processed_peaks 作为输入基础。")