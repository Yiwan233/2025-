# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题1：贡献者人数识别

代码名称: P1_V2.4_Complete_Stutter_Workflow
版本: 2.4
日期: 2025-05-11
描述: 整合了所有必要的全局参数定义（包括STUTTER_CV_HS_GLOBAL）、
      辅助函数、数据加载、Stutter概率化评估、以及特征工程的完整工作流。
      修正了之前版本中可能因变量未定义导致的NameError。
      输出包含真实等位基因置信度(CTA)的处理后峰数据(df_processed_peaks)
      和基于此计算的新特征集(df_features_v2_4)。
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

# --- 配置与环境设置 (版本 2.4) ---
print("--- 脚本初始化与配置 (版本 2.4) ---")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("INFO: Matplotlib 中文字体尝试设置为 'SimHei'.")
except Exception as e_font:
    print(f"警告: 设置 Matplotlib 中文字体 'SimHei' 失败: {e_font}.")

DATA_DIR = './'
file_path_prob1 = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv') # 确保这是您附件1的文件名
PLOTS_DIR = os.path.join(DATA_DIR, 'prob1_plots_v2.4_final_eval')
processed_peaks_filename = os.path.join(DATA_DIR, 'prob1_processed_peaks_v2.4.csv')
stutter_debug_log_filename = os.path.join(DATA_DIR, 'prob1_stutter_debug_log_v2.4.csv')
feature_filename_prob1_v2_4 = os.path.join(DATA_DIR, 'prob1_features_v2.4.csv')

if not os.path.exists(PLOTS_DIR):
    try:
        os.makedirs(PLOTS_DIR)
        print(f"已创建绘图保存目录: {PLOTS_DIR}")
    except OSError as e_dir:
        PLOTS_DIR = DATA_DIR

# --- 中文翻译映射表定义 (版本 2.4) ---
COLUMN_TRANSLATION_MAP = {
    'Sample File': '样本文件', 'Marker': '标记', 'Dye': '染料',
    'Allele': '等位基因', 'Allele_Numeric': '数值型等位基因',
    'Size': '片段大小(bp)', 'Height': '校正后峰高(RFU)', 'Original_Height': '原始峰高(RFU)',
    'CTA': '真实等位基因置信度', 'Is_Stutter_Suspect': '高度可疑Stutter',
    'Stutter_Score_N_Minus_1': '作为n-1 Stutter的得分', # 注意这里与V2.3的列名Stutter_Score_as_N_minus_1保持一致
    'NoC_True': '真实贡献人数', 'max_allele_per_sample': '样本内最大有效等位基因数',
    'total_alleles_per_sample': '样本内总有效等位基因数', 'avg_alleles_per_marker': '每标记平均有效等位基因数',
    'markers_gt2_alleles': '有效等位基因数>2的标记数', 'markers_gt3_alleles': '有效等位基因数>3的标记数',
    'markers_gt4_alleles': '有效等位基因数>4的标记数', 'avg_peak_height': '有效等位基因平均峰高',
    'std_peak_height': '有效等位基因峰高标准差', 'baseline_pred': '基线模型预测NoC'
}
DESCRIBE_INDEX_TRANSLATION_MAP = {
    'count': '计数', 'mean': '均值', 'std': '标准差', 'min': '最小值',
    '25%': '25%分位数', '50%': '中位数(50%)', '75%': '75%分位数', 'max': '最大值'
}
CLASSIFICATION_REPORT_METRICS_MAP = {'precision': '精确率', 'recall': '召回率', 'f1-score': 'F1分数', 'support': '样本数'}
CLASSIFICATION_REPORT_AVG_MAP = {'accuracy': '准确率(整体)', 'macro avg': '宏平均', 'weighted avg': '加权平均'}

# --- 全局参数定义 (版本 2.4) ---
print(f"\n--- 全局参数定义 (版本 2.4) ---")
SATURATION_THRESHOLD = 30000.0
SIZE_TOLERANCE_BP = 0.5
STUTTER_CV_HS_GLOBAL = 0.25 # **确保此变量已定义**
GLOBAL_AT = 50
DYE_AT_VALUES = {
    'B': 75, 'BLUE': 75, 'G': 101, 'GREEN': 101, 'Y': 60, 'YELLOW': 60,
    'R': 69, 'RED': 69, 'P': 56, 'PURPLE': 56, 'O': 50, 'ORANGE': 50,
    'UNKNOWN': GLOBAL_AT
}
MARKER_PARAMS = {
    # --- 请您在此处填充您整理的、最完整和最准确的 MARKER_PARAMS 字典 ---
    # 例如 (仅为部分示例，请务必用您最终确认的值替换):
    "AMEL":    {"L_repeat": 0, "Dye": "BLUE",   "n_minus_1_Stutter": {"SR_model_type": "N/A",             "SR_m": 0,         "SR_c": 0,         "RS_max_k": 0.0}},
    "D3S1358": {"L_repeat": 4, "Dye": "BLUE",   "n_minus_1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.09860,   "RS_max_k": 0.20}},
    "D1S1656": {"L_repeat": 4, "Dye": "BLUE",   "n_minus_1_Stutter": {"SR_model_type": "LUS Regression (fallback)",  "SR_m": 0.00604,   "SR_c": 0.00132,   "RS_max_k": 0.22}},
    # ... (为所有附件1中的Marker填充完整参数) ...
    "D8S1179": {"L_repeat": 4, "Dye": "RED",    "n_minus_1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.08246,   "RS_max_k": 0.3}},
    "D21S11":  {"L_repeat": 4, "Dye": "YELLOW", "n_minus_1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.08990,   "RS_max_k": 0.3}},
    "D7S820":  {"L_repeat": 4, "Dye": "YELLOW", "n_minus_1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.01048,  "SR_c": -0.05172,  "RS_max_k": 0.3}},
    "CSF1PO":  {"L_repeat": 4, "Dye": "GREEN",  "n_minus_1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.01144,  "SR_c": -0.05766,  "RS_max_k": 0.3}},
    "TH01":    {"L_repeat": 4, "Dye": "YELLOW", "n_minus_1_Stutter": {"SR_model_type": "LUS Regression (fallback)",  "SR_m": 0.00185,  "SR_c": 0.00801,   "RS_max_k": 0.10}},
    "D13S317": {"L_repeat": 4, "Dye": "BLUE",   "n_minus_1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.05638,   "RS_max_k": 0.3}},
    "D16S539": {"L_repeat": 4, "Dye": "GREEN",  "n_minus_1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.0118,   "SR_c": -0.0595,   "RS_max_k": 0.3}},
    "D2S1338": {"L_repeat": 4, "Dye": "GREEN",  "n_minus_1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.09165,   "RS_max_k": 0.3}},
    "D19S433": {"L_repeat": 4, "Dye": "RED",    "n_minus_1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.08044,   "RS_max_k": 0.3}},
    "VWA":     {"L_repeat": 4, "Dye": "YELLOW", "n_minus_1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.08928,   "RS_max_k": 0.20}},
    "TPOX":    {"L_repeat": 4, "Dye": "YELLOW", "n_minus_1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.00611,  "SR_c": -0.02772,  "RS_max_k": 0.15}},
    "D18S51":  {"L_repeat": 4, "Dye": "GREEN",  "n_minus_1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.00879,  "SR_c": -0.04708,  "RS_max_k": 0.3}},
    "D5S818":  {"L_repeat": 4, "Dye": "YELLOW", "n_minus_1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.07378,   "RS_max_k": 0.3}},
    "FGA":     {"L_repeat": 4, "Dye": "PURPLE", "n_minus_1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.08296,   "RS_max_k": 0.3}},
    "D2S441":  {"L_repeat": 4, "Dye": "BLUE",   "n_minus_1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.05268,   "RS_max_k": 0.15}},
    "D10S1248":{"L_repeat": 4, "Dye": "BLUE",   "n_minus_1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.01068,  "SR_c": -0.06292,  "RS_max_k": 0.3}},
    "Penta E": {"L_repeat": 5, "Dye": "BLUE",   "n_minus_1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.00398,  "SR_c": -0.01607,  "RS_max_k": 0.15}},
    "Penta D": {"L_repeat": 5, "Dye": "GREEN",  "n_minus_1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.01990,   "RS_max_k": 0.10}},
    "D12S391": {"L_repeat": 4, "Dye": "RED",    "n_minus_1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.01063,  "SR_c": -0.10533,  "RS_max_k": 0.3}},
    "SE33":    {"L_repeat": 4, "Dye": "RED",    "n_minus_1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.12304,   "RS_max_k": 0.20}},
    "D22S1045":{"L_repeat": 3, "Dye": "RED",    "n_minus_1_Stutter": {"SR_model_type": "LUS Regression (fallback)",  "SR_m": 0.01528,  "SR_c": -0.1354,   "RS_max_k": 0.25}},
    "DYS391":  {"L_repeat": 4, "Dye": "PURPLE", "n_minus_1_Stutter": {"SR_model_type": "N/A",             "SR_m": 0,         "SR_c": 0,         "RS_max_k": 0.0}},
    "DYS576":  {"L_repeat": 4, "Dye": "PURPLE", "n_minus_1_Stutter": {"SR_model_type": "N/A",             "SR_m": 0,         "SR_c": 0,         "RS_max_k": 0.0}},
    "DYS570":  {"L_repeat": 4, "Dye": "PURPLE", "n_minus_1_Stutter": {"SR_model_type": "N/A",             "SR_m": 0,         "SR_c": 0,         "RS_max_k": 0.0}},
}

# --- 辅助函数定义 (版本 2.4) ---
print(f"\n--- 辅助函数定义 (版本 2.4) ---")
def print_df_in_chinese(df_to_print, col_map=None, index_item_map=None, index_name_map=None, title="DataFrame 内容", float_format='{:.4f}'):
    print(f"\n{title}:")
    df_display = df_to_print.copy()
    if col_map: df_display.columns = [col_map.get(str(col), str(col)) for col in df_display.columns]
    if index_item_map: df_display.index = [index_item_map.get(str(idx), str(idx)) for idx in df_display.index]
    if index_name_map and df_display.index.name is not None: df_display.index.name = index_name_map.get(str(df_display.index.name), str(df_display.index.name))
    with pd.option_context('display.float_format', float_format.format, 'display.max_rows', None, 'display.max_columns', None, 'display.width', 120): print(df_display)

def extract_true_noc_v2_4(filename_str):
    filename_str = str(filename_str)
    match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', filename_str)
    if match:
        ids_list = [id_val for id_val in match.group(1).split('_') if id_val.isdigit()]
        return int(len(ids_list)) if len(ids_list) > 0 else np.nan
    return np.nan

def get_allele_numeric_v2_4(allele_val_str):
    try: return float(allele_val_str)
    except ValueError:
        if isinstance(allele_val_str, str):
            if allele_val_str.upper() == 'X': return -1.0 
            if allele_val_str.upper() == 'Y': return -2.0
        return np.nan

# --- 核心：峰处理与Stutter评估函数 (版本 2.4) ---
stutter_debug_log_list = []
def calculate_peak_confidence_v2_4(locus_peaks_df_input, marker_name, marker_params_dict,
                                 dye_at_dict, global_at_val, sat_threshold,
                                 size_tolerance, cv_hs_n_minus_1_global_param, # Renamed for clarity
                                 debug_log_list_param=None):
    locus_peaks_df = locus_peaks_df_input.copy()
    processed_peaks_output_cols = ['Allele', 'Allele_Numeric', 'Size', 'Height', 'Original_Height', 'CTA', 'Is_Stutter_Suspect', 'Stutter_Score_N_Minus_1']

    params = marker_params_dict.get(marker_name, 
                                    {'L_repeat': 0, 'Dye': locus_peaks_df['Dye'].iloc[0] if not locus_peaks_df.empty and 'Dye' in locus_peaks_df.columns else 'UNKNOWN', 
                                     'n_minus_1_Stutter': {'SR_model_type': 'N/A', 'RS_max_k':0.0}})
    l_repeat = params.get('L_repeat', 0)
    current_dye_from_data = locus_peaks_df['Dye'].iloc[0].upper() if not locus_peaks_df.empty and 'Dye' in locus_peaks_df.columns else 'UNKNOWN'
    current_dye_from_params = str(params.get('Dye', 'UNKNOWN')).upper()
    final_dye_for_at = current_dye_from_params if current_dye_from_params != 'UNKNOWN' else current_dye_from_data
    at_val = dye_at_dict.get(final_dye_for_at, global_at_val)

    candidate_peaks_list = []
    for _, peak_row in locus_peaks_df.iterrows():
        height_original = float(peak_row['Height'])
        height_adj = min(height_original, sat_threshold)
        if height_adj >= at_val:
            allele_numeric = get_allele_numeric_v2_4(peak_row['Allele'])
            candidate_peaks_list.append({
                'Allele': peak_row['Allele'], 'Allele_Numeric': allele_numeric,
                'Size': float(peak_row['Size']), 'Height': height_adj,
                'Original_Height': height_original, 'Stutter_Score_N_Minus_1': 0.0,
            })
    
    if not candidate_peaks_list:
        return pd.DataFrame(columns=processed_peaks_output_cols)

    peaks_df = pd.DataFrame(candidate_peaks_list).sort_values(by='Height', ascending=False).reset_index().rename(columns={'index': 'original_peak_index'})
    
    stutter_params_n_minus_1 = params.get('n_minus_1_Stutter', {})
    if l_repeat == 0 or stutter_params_n_minus_1.get('SR_model_type') == 'N/A' or pd.isna(stutter_params_n_minus_1.get('RS_max_k', np.nan)) or stutter_params_n_minus_1.get('RS_max_k', 0.0) == 0.0:
        peaks_df['CTA'] = 1.0
        peaks_df['Is_Stutter_Suspect'] = False
        return peaks_df[processed_peaks_output_cols]

    sr_max_n_minus_1 = stutter_params_n_minus_1.get('RS_max_k', 0.3)
    temp_max_stutter_scores_for_each_peak = [0.0] * len(peaks_df)

    for i in range(len(peaks_df)):
        pc_row = peaks_df.iloc[i]
        current_max_score_pc_is_stutter = 0.0
        for j in range(len(peaks_df)):
            if i == j: continue
            pp_row = peaks_df.iloc[j]
            if pc_row['Height'] >= pp_row['Height'] * 1.01 : continue

            is_pos_match_n_minus_1 = (abs((pp_row['Size'] - pc_row['Size']) - l_repeat) <= size_tolerance)
            
            debug_entry = None
            if debug_log_list_param is not None and is_pos_match_n_minus_1:
                 debug_entry = {'Marker': marker_name, 'Parent_Allele': pp_row['Allele'], 'Parent_Size': pp_row['Size'], 'Parent_Height_Corrected': pp_row['Height'],
                                'Candidate_Stutter_Allele': pc_row['Allele'], 'Candidate_Stutter_Size': pc_row['Size'], 'Candidate_Stutter_Height_Corrected': pc_row['Height'],
                                'Is_N_Minus_1_Position_Match': True, 'Observed_SR': np.nan, 'Marker_L_Repeat': l_repeat,
                                'Marker_SR_Max_N_Minus_1': sr_max_n_minus_1, 'Parent_Allele_Numeric': pp_row['Allele_Numeric'],
                                'Expected_SR': np.nan, 'Expected_Stutter_Height': np.nan, 'Assumed_CV_Hs': cv_hs_n_minus_1_global_param, # Use passed param
                                'Calculated_Sigma_Hs': np.nan, 'Z_Score': np.nan, 'Calculated_Stutter_Score_for_this_Pair': 0.0}

            if is_pos_match_n_minus_1:
                sr_obs = pc_row['Height'] / pp_row['Height'] if pp_row['Height'] > 1e-9 else np.inf
                if debug_entry: debug_entry['Observed_SR'] = sr_obs

                if sr_obs > sr_max_n_minus_1:
                    score_stutter_for_this_parent = 0.0
                else:
                    e_sr = 0.0
                    parent_an = pp_row['Allele_Numeric']
                    if pd.notna(parent_an) and parent_an >= 0:
                        model_type = stutter_params_n_minus_1.get('SR_model_type')
                        m_val = stutter_params_n_minus_1.get('SR_m', 0)
                        c_val = stutter_params_n_minus_1.get('SR_c', 0)
                        if model_type == 'Allele Regression' or model_type == 'LUS Regression (fallback)':
                            e_sr = m_val * parent_an + c_val
                        elif model_type == 'Allele Average':
                            e_sr = c_val
                    
                    e_sr = max(0.001, e_sr) 
                    e_hs = e_sr * pp_row['Height']
                    if debug_entry: 
                        debug_entry['Expected_SR'] = e_sr
                        debug_entry['Expected_Stutter_Height'] = e_hs
                    
                    current_score = 0.0
                    if e_hs > 1e-6 :
                        sigma_hs = cv_hs_n_minus_1_global_param * e_hs # Use passed param
                        if debug_entry: debug_entry['Calculated_Sigma_Hs'] = sigma_hs
                        if sigma_hs > 1e-6:
                            z_score = (pc_row['Height'] - e_hs) / sigma_hs
                            if debug_entry: debug_entry['Z_Score'] = z_score
                            current_score = exp(-0.5 * (z_score**2))
                            if sr_obs > e_sr * 1.8 and e_sr > 1e-6 : current_score *= 0.5 
                            if sr_obs > e_sr * 2.5 and e_sr > 1e-6 : current_score *= 0.1
                        else: 
                            current_score = 1.0 if abs(pc_row['Height'] - e_hs) < (0.001 * e_hs + 1e-6) else 0.0
                    elif pc_row['Height'] < 1e-6 : 
                        current_score = 1.0
                    
                    score_stutter_for_this_parent = current_score
                
                if debug_entry: 
                    debug_entry['Calculated_Stutter_Score_for_this_Pair'] = score_stutter_for_this_parent
                    debug_log_list_param.append(debug_entry)

                max_score_this_pc_is_stutter = max(max_score_this_pc_is_stutter, score_stutter_for_this_parent)
        
        temp_max_stutter_scores[i] = max_score_this_pc_is_stutter
        
    peaks_df['Stutter_Score_N_Minus_1'] = temp_max_stutter_scores
    peaks_df['CTA'] = 1.0 - peaks_df['Stutter_Score_N_Minus_1']
    peaks_df['CTA'] = peaks_df['CTA'].clip(lower=0.0, upper=1.0)
    peaks_df['Is_Stutter_Suspect'] = peaks_df['Stutter_Score_N_Minus_1'] >= 0.5

    return peaks_df[processed_peaks_output_cols]

# --- 主逻辑 ---
# 步骤 1: 数据加载与NoC提取 (您需要用V1.15中完整的Step1代码替换以下占位符)
print(f"\n--- 步骤 1：数据加载与 NoC 提取 (版本 2.4) ---")
df_prob1 = None; load_successful = False
try:
    df_prob1 = pd.read_csv(file_path_prob1, encoding='utf-8', sep=',', on_bad_lines='skip')
    if df_prob1.empty: raise ValueError("文件加载后为空或格式不正确。")
    load_successful = True; print(f"成功加载文件: '{file_path_prob1}'")
    required_cols_prob1 = ['Sample File', 'Marker', 'Dye'] + [f'Allele {i}' for i in range(1,3)] + [f'Size {i}' for i in range(1,3)] + [f'Height {i}' for i in range(1,3)]
    if not all(col in df_prob1.columns for col in required_cols_prob1):
        missing_cols = [col for col in required_cols_prob1 if col not in df_prob1.columns]
        raise ValueError(f"原始数据文件缺少核心列: {missing_cols}")
except FileNotFoundError: print(f"错误: 文件未找到 '{file_path_prob1}'"); exit()
except ValueError as ve: print(f"错误: {ve}"); exit()
except Exception as e: print(f"加载文件错误: {e}"); exit()

unique_files = df_prob1['Sample File'].dropna().unique()
if len(unique_files) == 0: print("错误: 未找到任何唯一的样本文件。"); exit()
noc_map = {filename: extract_true_noc_v2_4(filename) for filename in unique_files}
df_prob1['NoC_True_Mapped'] = df_prob1['Sample File'].map(noc_map)
df_prob1.dropna(subset=['NoC_True_Mapped'], inplace=True)
if df_prob1.empty: print("错误: 提取NoC并移除无效行后，数据为空。"); exit()
df_prob1['NoC_True'] = df_prob1['NoC_True_Mapped'].astype(int)
df_prob1.drop(columns=['NoC_True_Mapped'], inplace=True)
print(f"NoC 提取完成。数据维度: {df_prob1.shape}")
print("--- 步骤 1 完成 ---")

# --- 步骤 2: 峰识别与Stutter概率化评估 ---
print(f"\n--- 步骤 2: 峰识别与Stutter概率化评估 (版本 2.4) ---")
if df_prob1.empty: print("错误: df_prob1 为空。"); exit()
if not MARKER_PARAMS: print("错误: MARKER_PARAMS 字典为空。"); exit()

processed_peak_data_all_samples = []
stutter_debug_log_list = [] # 初始化全局日志列表
sample_files_processed_count = 0
unique_sample_files_total = df_prob1['Sample File'].nunique()

for sample_file_name, group_data_per_sample in df_prob1.groupby('Sample File'):
    sample_files_processed_count += 1
    if unique_sample_files_total > 0 :
        progress = sample_files_processed_count / unique_sample_files_total * 100
        print(f"正在处理样本: {sample_file_name} ({sample_files_processed_count}/{unique_sample_files_total} - {progress:.1f}%)", end='\r')
    
    sample_all_loci_processed_peaks = []
    for marker_name_actual, locus_data_from_groupby in group_data_per_sample.groupby('Marker'):
        current_locus_peaks_list = []
        if locus_data_from_groupby.empty: continue
        row_marker_data = locus_data_from_groupby.iloc[0]
        for i in range(1, 101):
            allele_val, size_val, height_val = row_marker_data.get(f'Allele {i}'), row_marker_data.get(f'Size {i}'), row_marker_data.get(f'Height {i}')
            if pd.notna(allele_val) and pd.notna(size_val) and pd.notna(height_val) and float(height_val) > 0:
                current_locus_peaks_list.append({'Allele': allele_val, 'Size': size_val, 'Height': height_val, 'Dye': str(row_marker_data.get('Dye', 'UNKNOWN')).upper()})
        
        if not current_locus_peaks_list: continue
        locus_peaks_for_filter_df = pd.DataFrame(current_locus_peaks_list)
        
        processed_locus_df = calculate_peak_confidence_v2_4(
            locus_peaks_for_filter_df, marker_name_actual, MARKER_PARAMS,
            DYE_AT_VALUES, GLOBAL_AT, SATURATION_THRESHOLD,
            SIZE_TOLERANCE_BP, STUTTER_CV_HS_GLOBAL, # 使用全局变量
            stutter_debug_log_list 
        )
        if not processed_locus_df.empty:
            processed_locus_df['Sample File'] = sample_file_name
            processed_locus_df['Marker'] = marker_name_actual
            sample_all_loci_processed_peaks.append(processed_locus_df)
    
    if sample_all_loci_processed_peaks:
        processed_peak_data_all_samples.extend(sample_all_loci_processed_peaks)
print("\n所有样本位点处理完成。")

if not processed_peak_data_all_samples: print("错误: 处理所有样本后，列表 processed_peak_data_all_samples 为空。"); exit()
df_processed_peaks = pd.concat(processed_peak_data_all_samples, ignore_index=True)
print(f"峰处理与Stutter评估完成。共处理 {sample_files_processed_count} 个独立样本。")
print(f"处理后的总峰条目数 (df_processed_peaks): {len(df_processed_peaks)}")
print_df_in_chinese(df_processed_peaks.head(), col_map=COLUMN_TRANSLATION_MAP, title="处理后峰数据示例 (df_processed_peaks)")

if stutter_debug_log_list:
    df_stutter_debug_log = pd.DataFrame(stutter_debug_log_list)
    try:
        df_stutter_debug_log.to_csv(stutter_debug_log_filename, index=False, encoding='utf-8-sig')
        print(f"Stutter评估详细日志已保存到: {stutter_debug_log_filename}")
    except Exception as e_save_debug_log: print(f"错误: 保存Stutter评估日志失败: {e_save_debug_log}")
else: print("没有生成Stutter评估日志。")

try:
    df_processed_peaks.to_csv(processed_peaks_filename, index=False, encoding='utf-8-sig')
    print(f"已处理的峰数据已保存到: {processed_peaks_filename}")
except Exception as e_save_peaks: print(f"错误: 保存处理后的峰数据失败: {e_save_peaks}")
print("--- 步骤 2 完成 ---")

# --- 步骤 3: 特征工程 (版本 2.4 - 基于新的 df_processed_peaks) ---
# ... (您需要将之前的步骤3代码复制到此处，并确保它使用 df_processed_peaks 和 CTA 列) ...
# print("--- 步骤 3 完成 ---")

# --- 步骤 4 & 5: 模型评估 ---
# ... (您需要将之前的步骤4和5代码复制到此处，并确保它使用新的特征数据框) ...
# print("--- 步骤 4 & 5 完成 ---")

print(f"\n脚本 {os.path.basename(__file__)} (版本 2.4) 执行完毕。")